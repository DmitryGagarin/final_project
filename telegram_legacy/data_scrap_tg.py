import asyncio
import sqlite3

import socks
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest

# Your credentials (already used to create the session)
api_id = 36020350
api_hash = '150dbc74b8eca6c4a906efaf678beb2f'

# Database setup
conn = sqlite3.connect('telegram_posts.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel TEXT,
        message_id INTEGER,
        date TEXT,
        text TEXT,
        UNIQUE(channel, message_id)
    )
''')
conn.commit()

# Public channels to scrape
channels = [
    'Bloomberg',
    'TraderTVLive',
    'macroresearch',
    'TheFinancialExpressOnline',
    'vanillafinancenews',
    'TradingGainX'
]


async def scrape_channel(client, channel_username):
    try:
        entity = await client.get_entity(channel_username)
        posts = await client(GetHistoryRequest(
            peer=entity,
            limit=150,
            offset_date=None,
            offset_id=0,
            max_id=0,
            min_id=0,
            add_offset=0,
            hash=0
        ))
        saved = 0
        for msg in posts.messages:
            if msg.message:
                cursor.execute('''
                    INSERT OR IGNORE INTO posts (channel, message_id, date, text)
                    VALUES (?, ?, ?, ?)
                ''', (channel_username, msg.id, msg.date.strftime('%Y-%m-%d %H:%M:%S'), msg.message))
                saved += 1
        conn.commit()
        print(f"{channel_username}: {saved} new posts")
    except Exception as e:
        print(f"{channel_username}: {e}")


async def main():
    # Load the saved session (no phone, QR, or password needed)
    client = TelegramClient(
        "user_session",  # must match the session file name
        api_id,
        api_hash,
        proxy=(socks.SOCKS5, "127.0.0.1", 10808)
    )

    await client.start()
    print("Logged in automatically via saved session")

    for channel in channels:
        await scrape_channel(client, channel)

    await client.disconnect()
    print("Done. All posts saved to telegram_posts.db")


if __name__ == '__main__':
    asyncio.run(main())
