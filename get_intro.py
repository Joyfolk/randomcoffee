import os
import asyncio
import argparse
import dotenv

from telethon import TelegramClient
from telethon.tl.types import PeerChannel

dotenv.load_dotenv()

API_ID = int(os.getenv('TELEGRAM_API_ID', 0))
API_HASH = os.getenv('TELEGRAM_API_HASH', '')
if not API_ID or not API_HASH:
    raise EnvironmentError("Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in environment variables or .env file")

parser = argparse.ArgumentParser(description='Fetch messages with a specific hashtag from a Telegram group chat or thread')
parser.add_argument('--group', '-g', required=True,
                    help='Group ID (e.g., -1001234567890) or username (e.g., myGroup)')
parser.add_argument('--thread', '-t', default=None,
                    help='Thread/topic ID to search within (optional)')
parser.add_argument('--tag', default='#Intro',
                    help='Hashtag to filter messages (default: #Intro)')
parser.add_argument('--session', '-s', default='session',
                    help='Name of the session file (default: session)')
parser.add_argument('--output', '-o', default='data/intros',
                    help='Directory to save intro files (default: data/intros)')
args = parser.parse_args()

GROUP_ID = args.group
THREAD_ID = args.thread
TAG = args.tag
SESSION_NAME = args.session
OUTPUT_DIR = args.output

async def fetch_intros():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        if GROUP_ID.isdigit():
            group_entity = PeerChannel(channel_id=int(GROUP_ID))
        elif GROUP_ID.startswith('-100') and GROUP_ID[4:].isdigit():
            group_entity = int(GROUP_ID)
        else:
            print(f"Error: Invalid group ID format. Please provide a valid numeric ID")
            return

        print(f"Connecting to group: {GROUP_ID}")

        try:
            await client.get_messages(group_entity, limit=1)
        except Exception as e:
            print(f"Error accessing group: {e}")
            print("Tips: Make sure you've joined the group and have the correct ID/username")
            return

        kwargs = {
            'limit': None
        }

        if THREAD_ID:
            print(f"Searching within thread: {THREAD_ID}")
            kwargs['reply_to'] = int(THREAD_ID)

        try:
            message_count = 0
            print(f"Searching for messages with {TAG}...")

            async for msg in client.iter_messages(group_entity, **kwargs):
                if msg.text and TAG in msg.text:
                    message_count += 1
                    username = msg.sender.username if msg.sender and msg.sender.username else str(msg.sender_id)

                    link = generate_link(msg)
                    text = msg.text.replace(TAG, '').strip()

                    filename = f"{username}.txt"
                    filepath = os.path.join(OUTPUT_DIR, filename)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(link + '\n')
                        f.write(text + '\n')

                    print(f"Saved intro for @{username} to {filepath}")

            if message_count == 0:
                print(f"No messages with {TAG} found in the specified group/thread.")
                print("Note: You must be a member of the group to access its messages.")
            else:
                print(f"{message_count} intros saved to {OUTPUT_DIR}.")

        except Exception as iteration_error:
            print(f"Error accessing messages: {iteration_error}")
            print("Troubleshooting tips:")
            print(" - Ensure you're a member of the group")
            print(" - Verify the group ID is correct (from t.me/c/GROUP_ID format)")
            print(" - Make sure your account has permission to view messages")

    except Exception as e:
        print(f"Error: {e}")

    await client.disconnect()


def generate_link(msg):
    raw_id = GROUP_ID.lstrip('-')
    if raw_id.startswith('100') and len(raw_id) > 3:
        raw_id = raw_id[3:]
    elif GROUP_ID.isdigit():
        raw_id = GROUP_ID
    link = f"https://t.me/c/{raw_id}/{msg.id}"
    if THREAD_ID:
        link += f"?thread={THREAD_ID}"
    return link


if __name__ == '__main__':
    asyncio.run(fetch_intros())
