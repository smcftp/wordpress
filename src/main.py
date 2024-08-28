import asyncio
import logging
import sys
from src.config import bot_tg
from src.handlers import register_handlers1, dp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

def main() -> None:
    try:
        register_handlers1(dp)
        asyncio.run(dp.start_polling(bot_tg))
    except Exception as e:
        logging.error(f"Ошибка в основной функции: {e}")

if __name__ == "__main__":
    main()
