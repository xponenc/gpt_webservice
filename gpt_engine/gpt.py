import re

from openai import AsyncOpenAI

from services.loggers import logger


async def generate_answer(query,  # запрос пользователя
                          db_index,  # векторная база знаний
                          k=5,  # используемое к-во чанков
                          verbose=True,  # выводить ли на экран выбранные чанки
                          model='gpt-4o-mini',
                          temp=0):
    system_prompt = """Ты - профессиональный специалист в области страхования. Ты работаешь в страховой компании Simba 
    и помогаешь клиентам компании в консультировании по вопросам страхования. На основании этой информации и
     предоставленных тебе выдержках руководящего документа ответь на вопросы клиента. Если на основании предоставленной
      информации ты не можешь ответить на вопрос, то ответь, что ты не знаешь ответа.
    В ответе не упоминай документ. Пользователь вообще не должен знать о документе.
    Отрывки документа: 
    """
    # Поиск чанков по векторной базе данных
    similar_documents = db_index.similarity_search(query, k=k)
    # Формирование текстового контента из выбранных чанков для модели
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'Отрывок документа № {i + 1}:\n' + doc.page_content
         for i, doc in enumerate(similar_documents)]))
    if verbose:
        print(message_content)  # печать на экран выбранных чанков

    user_prompt = query
    logger.info(f"Start create new game")
    logger.info(f"System prompt: {system_prompt + message_content}")
    logger.info(f"User prompt: {user_prompt}")

    # Используем await для асинхронного вызова
    gpt_answer, price, tokens_info = await get_response_openai(system_prompt + message_content, user_prompt)
    logger.info(tokens_info)

    # Возвращаем структуру и информацию о токенах
    return gpt_answer


# Функция подсчета токенов и стоимости
def tokens_count_and_price(completion, model='gpt-4o-mini'):
    if model == "gpt-4o-mini":
        input_price, output_price = 0.15 * 100, 0.60 * 100  # Переводим в рубли
    elif model == "gpt-4o":
        input_price, output_price = 5 * 100, 15 * 100  # Переводим в рубли
    else:
        raise ValueError("Неверная модель. Доступные модели: gpt-4o, gpt-4o-mini")

    price = (input_price * completion.usage.prompt_tokens /
             1e6 + output_price * completion.usage.completion_tokens / 1e6)
    return price, (f"\nTokens used: {completion.usage.prompt_tokens} + {completion.usage.completion_tokens} "
                   f"= {completion.usage.total_tokens}. Model: {model}, Price: {round(price, 2)} руб.")


# Функция для запроса и получения ответа от OpenAI с использованием chat моделей
async def get_response_openai(system_prompt, user_prompt, model='gpt-4o-mini', temp=0.0, max_tokens=4096):
    # Используем асинхронный вызов OpenAI API
    response = await AsyncOpenAI().chat.completions.create(
        model=model,
        temperature=temp,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    price, tokens_info = tokens_count_and_price(response, model)
    logger.info(f"GPT answer: {response}")
    return response.choices[
        0].message.content, price, tokens_info  # Возвращаем три значения: контент, цену и информацию о токенах
