# Telegram Group Intro Fetcher

## 


This script fetches messages with a specific hashtag from a Telegram group chat or thread.

## Setup

1. Obtain your Telegram API credentials by creating a new application on the [Telegram API development tools](https://my.telegram.org/apps) page.

2. Create a `.env` file based on `.env.example` with your Telegram API credentials or use environment variables:
   ```
   API_ID=your_api_id
   API_HASH=your_api_hash
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```
python get_intro.py --group GROUP_ID [--thread THREAD_ID] [--tag HASHTAG] [--output OUTPUT_DIR]
```

### Parameters

- `--group` or `-g`: Group ID or username. For numeric IDs, use the full format with -100 prefix (e.g., -1001234567890) or just the numeric part (e.g., 1234567890)
- `--thread` or `-t`: (Optional) Thread/topic ID to search within
- `--tag`: Hashtag to filter messages (default: #Intro)
- `--session` or `-s`: Name of the session file (default: session)
- `--output` or `-o`: Directory to save intro files (default: data/intros)

## Important Notes

- You must be a member of the group to fetch messages
- You can specify the group in two ways:
  - By username: `--group myGroup` 
  - By numeric ID: 
    - Full format: `--group -1001234567890` 
    - Short format: `--group 1234567890` (from t.me/c/1234567890/2)
- Thread ID refers to the message ID that started the thread/topic
  - In modern Telegram groups, threads/topics appear at the top of the group
  - The thread ID is visible in the URL when viewing the thread
- If you encounter permission errors:
  - Make sure you've joined the group through your Telegram account
  - Try interacting with the group first through the official Telegram app
  - Verify that your account has the necessary permissions to view messages

## Troubleshooting

If you're getting errors about not finding the entity:

1. Make sure you've joined the group through your Telegram account
2. Try using the official Telegram app to view the group first
3. Verify the group ID by checking the URL when viewing the group in a browser
4. For private groups, you must be a member and have the necessary permissions

## Behavior with Multiple Intro Posts

If multiple posts from the same user contain the intro tag:

- Only the most recent post will be saved
- The script saves messages to files named after the username (e.g., `username.txt`)
- When processing multiple tagged posts from the same user, each new post overwrites the previous one
- Since messages are processed from newest to oldest, only the newest intro post from each user is preserved
