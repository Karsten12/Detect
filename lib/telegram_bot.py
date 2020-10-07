import telegram
from telegram.ext import Updater, CommandHandler
import cv2
import io
import lib.utils as utils


def vacation_mode(update, context):
    """Enable/disable vacation mode when the command /vacation is issued"""
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, hello")
    # TODO


def send_frame(update, context):
    """Send a frame from all cameras, when the command /frame is issued"""
    # TODO
    # send_media_group(chat_id=update.effective_chat.id)


def stats(update, context):
    """Send some brief stats, when the command /stats is issued"""
    # TODO


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


def poll(telegram_token):
    print("Starting telegram bot")
    # Handle all the user commands
    updater = Updater(token=telegram_token, use_context=True)
    dp = updater.dispatcher
    # Add handlers to respond to each command from a user
    dp.add_handler(CommandHandler("vacation", vacation_mode))
    dp.add_handler(CommandHandler("frame", send_frame))
    dp.add_handler(CommandHandler("stats", stats))

    # Poll/check for commands from the user
    updater.start_polling()
    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    # updater.idle()


if __name__ == "__main__":
    config_dict = utils.load_config()

    telegram_token = config_dict["telegram_token"]
    people = config_dict["people"]  # Dict of people
    karsten = people["Karsten"]

    bot = telegram.Bot(telegram_token)

    send_message(bot, karsten, "123")
    # send_photo(bot, karsten, io_buf)
