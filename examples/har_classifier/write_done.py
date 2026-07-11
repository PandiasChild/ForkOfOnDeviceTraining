import json
import requests
def send_notification(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        requests.post(url, data=data)
    except:
        None
def main():
    with open('telegram_bot.json', 'r') as f:
        telegram_bot = json.load(f)

    bot_token = telegram_bot.get("BOT_TOKEN", {})
    chat_id = telegram_bot.get("CHAT_ID", {})
    message = f"TRAINING DONE!!!\n"
    send_notification(bot_token, chat_id, message)

if __name__ == "__main__":
    main()