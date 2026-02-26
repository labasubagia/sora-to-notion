# Sora CLI Tool

A command-line tool for managing and backing up AI-generated images from **Sora** and **ChatGPT**, with integration to store them in **Notion**.

## Features

- 📥 Download generated images from Sora and ChatGPT
- 📝 Embed prompts as PNG metadata
- 📤 Upload images to Notion database
- 🗑️ Cleanup/trash generations after upload
- ⚡ Concurrent downloads and uploads for speed
- 🔄 Automatic retry with exponential backoff for failed requests

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd sora
```

2. **Install dependencies**

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. **Configure environment variables**

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Notion API credentials
NOTION_API_KEY=your_notion_integration_token
NOTION_DATABASE_ID=your_database_id

# ChatGPT/Sora credentials (inspect browser network tab)
CHATGPT_AUTHORIZATION_TOKEN=your_auth_token
CHATGPT_USER_AGENT=your_user_agent
CHATGPT_COOKIE_STRING_BASE64=base64_encoded_cookie_string
```

> **Security Note**: Never commit your `.env` file to version control. The `.gitignore` is configured to exclude it.

## Usage

### CLI Commands

Run the CLI with `--help` to see all available commands:

```bash
python main.py --help
```

#### Upload Sora Generations to Notion

```bash
python main.py sora-upload-to-notion \
  --image-folder sora_images \
  --db-id YOUR_NOTION_DB_ID \
  --upload-to-notion true \
  --trash-in-sora false \
  --remove-in-sora false \
  --dataset sora_backup_2024.csv
```

**Options:**
- `--image-folder`: Folder to store downloaded images (default: `sora_images`)
- `--db-id`: Notion database ID
- `--upload-to-notion`: Whether to upload to Notion (default: `true`)
- `--trash-in-sora`: Move uploaded items to trash (default: `false`)
- `--remove-in-sora`: Permanently delete uploaded items (default: `false`)
- `--dataset`: CSV file to save generation metadata

#### Upload ChatGPT Image Generations to Notion

```bash
python main.py chatgpt-upload-to-notion \
  --image-folder chatgpt_images \
  --db-id YOUR_NOTION_DB_ID \
  --limit 100
```

**Options:**
- `--image-folder`: Folder to store downloaded images (default: `chatgpt_images`)
- `--db-id`: Notion database ID
- `--limit`: Maximum number of generations to process (default: `100`)
- `--remove-in-chatgpt`: Delete conversations after upload (default: `false`)

#### Cleanup Commands

**Clean up trashed Sora generations:**
```bash
python main.py sora-cleanup-trash --dataset sora_trash_backup.csv
```

**Delete empty Sora tasks:**
```bash
python main.py sora-cleanup-tasks
```

**Clean output directory:**
```bash
python main.py clean-output-path
```

## Project Structure

```
sora/
├── main.py          # CLI entry point with Typer commands
├── sora.py          # Sora API client and operations
├── chatgpt.py       # ChatGPT API client and operations
├── notion.py        # Notion API client and operations
├── img.py           # Image processing (add prompts to PNG metadata)
├── util.py          # Shared utilities (retry logic, path handling, etc.)
├── .env.example     # Environment variable template
└── output/          # Default output directory for downloads
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NOTION_API_KEY` | Notion integration token | For Notion operations |
| `NOTION_DATABASE_ID` | Target Notion database ID | For Notion operations |
| `CHATGPT_AUTHORIZATION_TOKEN` | ChatGPT/Sora auth token | For Sora/ChatGPT operations |
| `CHATGPT_USER_AGENT` | User-Agent header | For Sora/ChatGPT operations |
| `CHATGPT_COOKIE_STRING_BASE64` | Base64-encoded cookies | For ChatGPT operations |

### How to Get ChatGPT/Sora Credentials

1. Open your browser's Developer Tools (F12)
2. Go to the Network tab
3. Visit [Sora](https://sora.chatgpt.com) or [ChatGPT](https://chatgpt.com)
4. Look for XHR/fetch requests
5. Inspect request headers to find:
   - `Authorization` token
   - `User-Agent`
   - `Cookie` string (encode to base64 for `CHATGPT_COOKIE_STRING_BASE64`)

## Notion Database Setup

Create a Notion database with the following properties:

| Property Name | Type | Description |
|---------------|------|-------------|
| `Name` | Title | Image filename |
| `Image` | Files | The uploaded image |
| `Prompt` | Rich Text | Generation prompt |
| `Model` | Select | Model used (e.g., "Sora") |
| `Face` | Select | Face option used |

## Advanced Usage

### Reading PNG Metadata

After processing, prompts are embedded in PNG files. Use `exiftool` to read them:

```bash
exiftool -Prompt output/sora_images/abc123.png
```

### Dataset CSV Format

Generated CSV files contain:
- `created_at`: Generation timestamp
- `id`: Generation/image ID
- `task_id`/`conversation_id`: Parent task/conversation
- `url`: Original image URL
- `prompt`: Generation prompt

## Error Handling

- **HTTP 429 (Rate Limit)**: Automatic retry with exponential backoff
- **HTTP 5xx (Server Error)**: Automatic retry with exponential backoff
- **File Exists**: Skips already downloaded images
- **Network Errors**: Retries up to 5 times before failing

## Performance

Default concurrency settings:
- **Downloads**: 10 concurrent downloads
- **API Requests**: 10 concurrent requests
- **HTTP Timeout**: 30 seconds

Adjust these in `util.py` if needed.

## Testing

This project uses pytest for testing with a pure mocking approach (no real API calls in CI).

### Run Tests

```bash
# Run all tests
uv run pytest

# Run unit tests only (faster, no integration tests)
uv run pytest -m "not integration"

# Run with coverage report
uv run pytest --cov=. --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_util.py -v

# Run specific test function
uv run pytest tests/test_util.py::TestGetOutputPath::test_relative_path_allowed -v
```

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures and mock helpers
├── test_util.py         # Unit tests for utility functions (26 tests)
├── test_img.py          # Unit tests for image processing (9 tests)
├── test_notion.py       # Integration tests with mocking (11 tests)
├── test_chatgpt.py      # Integration tests with mocking (10 tests)
├── test_sora.py         # Integration tests with mocking (11 tests)
└── test_main.py         # CLI command tests (15 tests)
```

### Test Markers

- `@pytest.mark.integration` - Integration tests using mocked API responses
- `@pytest.mark.smoke` - Tests that require real API access (skip in CI)
- `@pytest.mark.slow` - Slow-running tests

### CI/CD

GitHub Actions runs on every push and pull request:

- **Lint**: `ruff check .`
- **Unit Tests**: pytest with HTML coverage report
- **Integration Tests**: Mocked API tests
- **Build**: Package build verification

Coverage reports are uploaded as GitHub artifacts (available for 7 days).
Download `coverage-report.zip` from the workflow run and open `htmlcov/index.html` in your browser.

## Troubleshooting

### "Missing required environment variables"

Ensure your `.env` file is properly configured and all required variables are set.

### "Notion database ID must be a valid ID"

The database ID should be a 32-character string (with or without hyphens).

### Images not uploading to Notion

1. Check that your Notion integration has access to the database
2. Verify the database ID is correct
3. Ensure the database has the required properties

### API rate limiting

The tool includes automatic retry logic, but if you hit rate limits frequently, consider reducing the `--limit` parameter.

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linting:
   ```bash
   uv run pytest -m "not integration"  # Run unit tests
   uv run ruff check .                 # Run linter
   ```
5. Commit your changes: `git commit -m "Add my feature"`
6. Push to the branch: `git push origin feature/my-feature`
7. Submit a pull request

### Testing Requirements

- All new features should include unit tests
- API interactions should be mocked in tests
- Maintain >80% code coverage
- All tests must pass before merging
