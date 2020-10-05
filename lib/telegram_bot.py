import telegram
from telegram.ext import Updater, CommandHandler
import cv2
import io
import lib.utils as utils


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, hello")


def send_message(bot, chat_id, message, silent=False):
    bot.send_message(chat_id=chat_id, text=message, disable_notification=silent)


def send_photo(bot, chat_id, photo, silent=False):
    bot.send_photo(chat_id=chat_id, photo=photo, disable_notification=silent)


def send_video(bot, chat_id, video, silent=False):
    bot.send_message(chat_id=chat_id, text=message, disable_notification=silent)


def idk(frame=None, thresh=None):
    text = "Motion detected at {}".format(datetime.now().strftime("%m-%d-%Y--%H-%M-%S"))
    # Convert cv2 image to format usable by telegram
    img = cv2.imread("images/sample_image2.png")
    # img = get_padding_detection(frame, thresh)
    buffer = cv2.imencode(".png", img)[1]
    io_buf = io.BytesIO(buffer)

    bot.send_photo(
        chat_id=chat_id, photo=io_buf, caption=text, disable_notification=silent
    )


if __name__ == "__main__":
    config_dict = utils.load_config()

    telegram_token = config_dict["telegram_token"]
    people = config_dict["people"]  # Dict of people
    karsten = people["Karsten"]

    bot = telegram.Bot(telegram_token)

    send_message(bot, karsten, "123")
    # send_photo(bot, karsten, io_buf)

    # updater = Updater(token=telegram_token, use_context=True)
    # dispatcher = updater.dispatcher

    # start_handler = CommandHandler('start', start)
    # dispatcher.add_handler(start_handler)

    # updater.start_polling()
