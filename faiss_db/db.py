# импорт библиотек
import re
from glob import glob

import requests
import tiktoken
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv  # работа с переменными окружения
import os  # взаимодействие с операционной системой

from langchain_core.embeddings import Embeddings
from openai import OpenAI  # взаимодействие с OpenAI API
from langchain.text_splitter import CharacterTextSplitter  # библиотека langchain
from langchain.docstore.document import Document  # объект класса Document
from langchain_community.vectorstores import FAISS  # модуль для работы с векторной базой FAISS
from langchain_openai import OpenAIEmbeddings  # класс для работы с ветроной базой

from services.loggers import logger

# получим переменные окружения из .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class DataText:
    def __init__(self):
        self.__text = None

    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, text):
        self.__text = text

    def download_google_doc(self, url: str) -> None:
        """Функция для загрузки документа Docx по ссылке из google драйв
        :param url: ссылка на расшаренный google документе
        :return:
        """
        match_ = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
        if not match_:
            logger.error(f"Invalid Google Docs URL {url}")
            raise ValueError("Invalid Google Docs URL")

        doc_id = match_.group(1)
        download_url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'
        logger.info(f"Started download Document {doc_id} from URL {download_url}")
        response = requests.get(download_url, stream=True)
        if response.status_code != 200:
            logger.error(f"Failed to download the document {doc_id} from {url},"
                         f" response: {response.status_code} {response}")
            raise RuntimeError("Failed to download the document")
        logger.info(f"Document {doc_id} download completed successfully from URL {download_url}")
        self.__text = response.text

    @staticmethod
    def clean_text(text: str = ""):
        """
        Функция для очистки текста
        :param text: очищаемый текст
        :return:
        """
        # Удаление заголовков и подзаголовков
        text = re.sub(r"\* .+\n", "", text)

        # Удаление телефонов и электронных адресов
        # text = re.sub(r"\+\d+ \(\d+\) \d+-\d+-\d+", "", text)
        # text = re.sub(r"\w+@\w+\.\w+", "", text)

        # Удаление разделительных линий и специальных символов
        text = re.sub(r"__+", "", text)

        # Удаление ссылок и инструкций JavaScript
        text = re.sub(r"Нажимая кнопку .+\n", "", text)
        text = re.sub(r"Пожалуйста, включите JavaScript .+\n", "", text)

        # Удаление строк типа "Оценка квартиры для [Название Банка] Подробнее"
        text = re.sub(r"Оценка квартиры для .+ Подробнее \n", "", text)

        # Удаление строк, содержащих "Заказать звонок"
        text = re.sub(r".*Заказать звонок.*\n", "", text)

        # Очистка текста от лишних пробелов и переводов строк
        text = re.sub(r"\n\s*\n", "\n", text)
        return text

    def replace_headers_with_markdown(self):
        pattern = r'(\d+\.+\s[А-ЯЁA-Z0-9\s]+)\s*(?=\n|$)'
        pattern = r'(\d+\.+\s[А-ЯЁA-Z0-9\s]+|Приложение\s№\s*\d+)\s*(?=\n|$)'

        # Функция для замены заголовков на формат Markdown
        def replace(match):
            header = match.group(1)  # Заголовок (например, "3. ОБЪЕКТ СТРАХОВАНИЯ")
            # Преобразуем заголовок в формат Markdown (например, "### 3. ОБЪЕКТ СТРАХОВАНИЯ")
            return f'### {header.strip()}'

        # Заменяем все совпадения в тексте
        new_text = re.sub(pattern, replace, self.__text, flags=re.MULTILINE)
        self.__text = new_text


class Chunks:

    def __init__(self):
        self.__chunks = None

    @property
    def chunks(self):
        return self.__chunks

    @chunks.setter
    def chunks(self, chunks):
        self.__chunks = chunks

    def show_chunks(self):
        for chunk in self.__chunks:
            print(chunk)
            print("\n ########## \n")

    def split_markdown_text(self, markdown_text: str,
                            strip_headers=False):
        """MarkdownHeaderTextSplitter Формируем chunks в формат LangChain Document из текста с Markdown разметкой
        :param markdown_text: Текст с разметкой Markdown
        :param strip_headers: НЕ удалять заголовки под '#..' из page_content
        :return:
        """

        logger.info(f"Started splitting into chunks using the method MarkdownHeaderTextSplitter"
                    f" for {markdown_text[:30]}")

        # Удалить пустые строки и лишние пробелы
        markdown_text = re.sub(r' {1,}', ' ', re.sub(r'\n\s*\n', '\n', markdown_text))
        headers_to_split_on = [("#", "Header 1"),
                               ("##", "Header 2"),
                               ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on,
                                                       strip_headers=strip_headers)
        chunks = markdown_splitter.split_text(markdown_text)
        self.__chunks = chunks
        logger.info(f"Completed splitting into chunks using the method MarkdownHeaderTextSplitter"
                    f" for {markdown_text[:30]}")
        self.audit_chunks()
        return self.__chunks

    def split_hard_text(self, text: str,
                        separators=None,
                        chunk_size=3000,
                        chunk_overlap=100):
        """RecursiveCharacterTextSplitter Формируем чанки в формат LangChain Document из текста по количеству символов
        :param separators: массив из строк разделителей, по которым будет делиться текст
        :param text: разделяемый текст
        :param chunk_size: Ограничение к-ва символов в чанке
        :param chunk_overlap: Количество символов перекрытия в чанке
        :return:
        """

        if separators is None:
            separators = ["\n\n", "\n", ". ", ", ", " ", ""]
        logger.info(fr"Started splitting into chunks using the method RecursiveCharacterTextSplitter "
                    f"with separators = {separators}, chunk_size = {chunk_size},"
                    f" chunk_overlap = {chunk_overlap}"
                    f" for {text[:30]}")
        # Удалить пустые строки и лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            # length_function=len
        )
        chunks = text_splitter.split_text(text)
        chunks = [Document(page_content=chunk) for chunk in chunks]
        self.__chunks = chunks
        logger.info(f"Completed splitting into chunks using the method RecursiveCharacterTextSplitter "
                    f"with separators = {separators}, chunk_size = {chunk_size},"
                    f" chunk_overlap = {chunk_overlap}"
                    f" for {text[:30]}")
        self.audit_chunks()

    def split_text(self, text: str,
                   separator=" ",
                   chunk_size=2000,  # Ограничение к-ва символов в чанке
                   chunk_overlap=200):  # к-во символов перекрытия в чанке
        """CharacterTextSplitter Формируем чанки из текста по количеству символов
        :param text: разделяемый текст
        :param separator: Разделитель
        :param chunk_size: Ограничение к-ва символов в чанке
        :param chunk_overlap: Количество символов перекрытия в чанке
        :return:
        """

        logger.info(f"Started splitting into chunks using the method CharacterTextSplitter "
                    f"with separators = '{separator}', chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}"
                    f" for {text[:30]}")
        # Удалить пустые строки и лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                         chunk_overlap=chunk_overlap,
                                         separator=separator)
        chunks = splitter.split_text(text)
        self.__chunks = chunks
        logger.info(f"Started splitting into chunks using the method CharacterTextSplitter "
                    f"with separators = '{separator}', chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}"
                    f" for {text[:30]}")
        self.audit_chunks()

    def audit_chunks(self, model='gpt-4o-mini'):
        """Функция возвращает количество токенов в self.__chunks в зависимости от используемой модели
        :param model: Модель для расчета количества токенов чанке
        :return:"""
        try:  # для формата чанков LangChain Document
            chunk_token_counts = [
                self.num_tokens_from_string(string=chunk.page_content, model=model) for chunk in self.__chunks
            ]
        except:  # для текстового формата чанков
            chunk_token_counts = [
                self.num_tokens_from_string(string=chunk, model=model) for chunk in self.__chunks
            ]
            print(chunk_token_counts)
        formatted_chunk_token_counts = {}
        for token_count in chunk_token_counts:
            formatted_chunk_token_counts.setdefault(token_count, 0)
            formatted_chunk_token_counts[token_count] += 1
        formatted_chunk_token_counts = dict(sorted(formatted_chunk_token_counts.items(),
                                                   key=lambda x: x[1], reverse=False))

        print("\nОбщее количество чанков: ", len(self.__chunks))
        print("Анализ распределения количества токенов в чанках:\n\tКол-во\tДлина чанка в токенах\n")
        for token_len, counter in formatted_chunk_token_counts.items():
            print(f"\t{counter}\t{token_len}")
        logger.info(f"Audit of splitting chunks: Number of chunks {len(self.__chunks)},"
                    f"Distribution of chunks by the number of tokens: {formatted_chunk_token_counts}")

    @staticmethod
    def num_tokens_from_string(string: str, model='gpt-4o-mini') -> int:
        """Получает количество токенов в чанке
        :param string: Текстовые данные чанка
        :param model: Модель для расчета количества токенов чанке
        :return:"""
        # Получаем имя кодировки для указанной модели
        encoding_name = tiktoken.encoding_for_model(model).name
        # Получаем объект кодировки на основе имени кодировки
        encoding = tiktoken.get_encoding(encoding_name)
        # Кодируем строку и вычисляем количество токенов
        num_tokens = len(encoding.encode(string))
        # Возвращаем количество токенов
        return num_tokens

    def custom_split_chunks(self):
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            chunk_size=3000,
            chunk_overlap=0
        )
        source_chunks = []
        for chunk in self.__chunks:
            if self.num_tokens_from_string(string=chunk.page_content) < 3000:
                source_chunks.append(chunk)
                continue
            for sub_chunk in splitter.split_text(chunk.page_content):
                source_chunks.append(Document(page_content=sub_chunk, metadata=chunk.metadata))
        self.__chunks = source_chunks


class FaissVectorDataBase:

    # МЕТОД: инициализация
    def __init__(self, db_name: str):
        self.__name = db_name
        self.__system = '''
                   Ты-консультант в компании Simble, ответь на вопрос клиента на основе документа с информацией.
                   Не придумывай ничего от себя, отвечай максимально по документу.
                   Не упоминай Документ с информацией для ответа клиенту.
                   Клиент ничего не должен знать про Документ с информацией для ответа клиенту            
               '''
        self.__urls = {}
        self.__db = None
        logger.info(f"Created {self.__class__.__name__} {self.__name}")

    @property
    def urls(self):
        return self.__urls

    @urls.setter
    def urls(self, url):
        self.__urls[len(self.__urls)] = url

    @property
    def db_vector(self):
        return self.__db

    @db_vector.setter
    def db_vector(self, db_vector):
        self.__db = db_vector

    def create_faiss_db(self,
                        chunks_documents: list[Document],
                        embeddings_model: Embeddings,
                        db_folder_path: str,
                        index_name: str,
                        rewrite_db: bool = False):
        """Создание индексной (векторной) базы из чанков в формате LangChain Document и сохранение на диск

        :param chunks_documents: список LangChain Documents
        :param embeddings_model: модель, которая будет использоваться для преобразования чанков в вектора
        :param db_folder_path: путь к директории сохранения модели
        :param index_name: названия файлов с
        :param rewrite_db:
        :return:
        """
        # embeddings_model = OpenAIEmbeddings()
        # Создание FAISS vector store
        store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_folder_path)
        # store_path = os.path.join(os.pardir, BASE_CONTENT_FOLDER, db_folder_path)  # Версия для колаба
        faiss_file = os.path.join(store_path, index_name + ".faiss")
        if not os.path.exists(faiss_file) or rewrite_db:
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            faiss_db_index = FAISS.from_documents(chunks_documents, embeddings_model)
            faiss_db_index.save_local(folder_path=store_path, index_name=index_name)
        else:
            faiss_db_index = FAISS.load_local(folder_path=store_path,
                                              index_name=index_name,
                                              embeddings=embeddings_model,
                                              allow_dangerous_deserialization=True)
        self.__db = faiss_db_index

    @staticmethod
    def load_db_vector(
            db_folder_path,
            index_name,
            embeddings_model):
        """Загрузка индексной (векторной) базы FAISS из файла
        :param db_folder_path:
        :param index_name:
        :param embeddings_model:
        :return:
        """
        store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_folder_path)
        return FAISS.load_local(
            allow_dangerous_deserialization=True,
            embeddings=embeddings_model,
            folder_path=store_path,
            index_name=index_name)

    def get_dbs_vector(self,
                       path_dbs: str,
                       embeddings_model: Embeddings):
        """
        Объединение сохраненных FAISS БД в одну
        :param path_dbs: директория с сохраненными БД
        :param embeddings_model:
        :return:
        """
        # Создание пустой векторной базы -----------------------------------------------------
        db_index_new = FAISS.from_documents([Document(page_content='', metadata={})],
                                            embeddings_model)  # FAISS.from_documents
        # или для текстового формата чанков:
        # db_index_new = FAISS.from_texts([''], embeddings_model)  # FAISS.from_texts

        for file in glob(f'{path_dbs}{os.sep}*.faiss'):
            index_name = file.split('/')[-1].split('.')[0]  # Имя векторной базы на диске
            # Загрузка ранее сохраненной векторной базы
            db_index_loaded = self.load_db_vector(path_dbs, index_name, embeddings_model)
            print(f'Векторная база: "{index_name}" загружена')
            # Слияние баз в одну db_index_new ------------------
            db_index_new.merge_from(db_index_loaded)
            # --------------------------------------------------
        self.__db = db_index_new


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()

    alfa_insurance_faiss_db = FaissVectorDataBase(db_name="alfa_insurance")

    doc_url = "https://docs.google.com/document/d/11MU3SnVbwL_rM-5fIC14Lc3XnbAV4rY1Zd_kpcMuH4Y"
    alfa_insurance_faiss_db.urls = doc_url
    doc_text = DataText()
    doc_text.download_google_doc(url=doc_url)
    doc_text.replace_headers_with_markdown()
    doc_text.text = doc_text.clean_text(doc_text.text)

    doc_chunks = Chunks()
    doc_chunks.split_markdown_text(markdown_text=doc_text.text)
    doc_chunks.audit_chunks()
    # doc_chunks.show_chunks()
    # doc_chunks.custom_split_chunks()
    # doc_chunks.audit_chunks()
    # doc_chunks.show_chunks()

    alfa_insurance_faiss_db.create_faiss_db(chunks_documents=doc_chunks.chunks,
                                            embeddings_model=embeddings,
                                            db_folder_path="saved_db",
                                            index_name="alfa_insurance_faiss_db")
