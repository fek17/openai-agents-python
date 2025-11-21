# Multi-Agent Email Contact Scraper

## Overview

This is an improved email contact scraper that uses a multi-agent architecture to find senior executives and their email addresses at companies.

## Key Improvements Over Single-Agent Version

### 1. **Specialized Agents**

Instead of one agent trying to do everything, we now have:

- **EmailFormatAgent**: Focused solely on discovering email format patterns
  - Searches for company email formats
  - Looks for patterns in company communications
  - Returns structured format templates

- **ExecutiveSearchAgent**: Focused solely on finding senior leadership
  - Searches for C-suite executives, VPs, Directors
  - Filters out non-senior roles automatically
  - Returns structured person data

- **CoordinatorAgent**: Orchestrates the entire workflow
  - Calls both specialized agents
  - Combines their results
  - Generates and exports the final contact list

### 2. **Better Separation of Concerns**

Each agent has:
- Clear, focused instructions
- Specific search strategies
- Well-defined output formats
- Single responsibility

### 3. **Parallel Execution Capability**

The architecture supports running both agents in parallel, which can:
- Reduce total execution time
- Make better use of API rate limits
- Improve overall throughput

### 4. **More Accurate Results**

Because each agent is specialized:
- Email format searches are more targeted
- Executive searches use better query strategies
- Less confusion about what each step should do
- Clearer output from each stage

## Architecture

```
┌─────────────────────────────────────────┐
│         CoordinatorAgent                │
│  (Orchestrates the full workflow)       │
└───────────┬─────────────────────────────┘
            │
            ├──────────────┬──────────────┐
            │              │              │
            ▼              ▼              │
   ┌────────────────┐ ┌──────────────────┐
   │ EmailFormat    │ │ Executive        │
   │ Agent          │ │ SearchAgent      │
   │                │ │                  │
   │ Finds email    │ │ Finds senior     │
   │ patterns       │ │ executives       │
   └────────┬───────┘ └────────┬─────────┘
            │                  │
            │                  │
            └──────────┬───────┘
                       ▼
              ┌────────────────┐
              │ generate_emails│
              │ (Tool)         │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ export_to_csv  │
              │ (Tool)         │
              └────────────────┘
```

## Usage

### Basic Usage

```python
from multi_agent_email_scraper import main

# Process a CSV or Excel file with companies
main(
    input_file="companies.csv",
    output_file="contacts_output.csv"
)
```

### Input File Format

Your input file should have at least 2 columns:
1. **Company Name** (required)
2. **Website URL** (optional but recommended)

Example CSV:
```csv
Company Name,Website
Acme Corporation,https://www.acme.com
TechStartup Inc,https://techstartup.io
```

Example Excel:
```
| Company Name      | Website                    |
|-------------------|----------------------------|
| Acme Corporation  | https://www.acme.com       |
| TechStartup Inc   | https://techstartup.io     |
```

### Single Company Processing

```python
from multi_agent_email_scraper import process_company

result_df, csv_path = process_company(
    company_name="Acme Corporation",
    company_domain="acme.com",
    company_website="https://www.acme.com",
    temp_file_prefix="temp",
    max_turns=30
)
```

## Configuration

### Adjusting Max Turns

If agents are hitting the turn limit, increase `max_turns`:

```python
main(
    input_file="companies.csv",
    output_file="contacts.csv",
    max_turns=40  # Default is 30
)
```

### Changing the Model

To use a different model, edit the agents in the code:

```python
email_format_agent = Agent(
    name="EmailFormatDiscovery",
    instructions="...",
    tools=[WebSearchTool()],
    model="gpt-4o",  # Change from gpt-4o-mini
)
```

### Customizing Seniority Filter

Edit the `is_senior_enough()` function to adjust which titles are included:

```python
def is_senior_enough(title: str) -> bool:
    """Check if a job title represents senior leadership."""
    # Modify these lists based on your criteria
    senior_keywords = [...]
    exclude_keywords = [...]
    # ...
```

## Output Format

The output CSV contains:

| Column      | Description                           |
|-------------|---------------------------------------|
| Company Name| The company name                      |
| First Name  | Executive's first name                |
| Last Name   | Executive's last name                 |
| Job Title   | Executive's job title                 |
| Email Guess | Generated email address               |

## How Email Generation Works

1. **Format Discovery**: The EmailFormatAgent searches for the company's email format patterns
2. **Executive Search**: The ExecutiveSearchAgent finds senior leadership names and titles
3. **Email Generation**: The `generate_emails` tool combines:
   - Name variants (first, last, initials)
   - Format patterns ({first}.{last}@domain.com, etc.)
   - Character transliteration (ä → ae, etc.)
4. **Validation**: Generated emails are validated against regex patterns
5. **Export**: Valid email guesses are exported to CSV

## Advanced: Async Usage

For better performance when processing multiple companies:

```python
import asyncio
from multi_agent_email_scraper import process_company_async

async def process_multiple():
    companies = [
        ("Acme Corp", "acme.com", "https://acme.com"),
        ("TechCo", "techco.io", "https://techco.io"),
    ]

    tasks = [
        process_company_async(name, domain, website)
        for name, domain, website in companies
    ]

    results = await asyncio.gather(*tasks)
    return results

# Run it
results = asyncio.run(process_multiple())
```

## Troubleshooting

### "Max turns exceeded"

- Increase `max_turns` parameter
- Simplify the coordinator agent instructions
- Check if agents are getting stuck in loops

### "No CSV created"

- Check if agents are finding any results
- Verify company names and domains are correct
- Look at console output for agent search results

### "Empty results"

- Try more well-known companies first
- Check if company domains are accessible
- Verify WebSearchTool is working correctly

### "Duplicate contacts"

The script automatically detects and removes duplicates based on first+last name combinations.

## Comparison: Single vs Multi-Agent

| Aspect | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| **Clarity** | One set of mixed instructions | Clear, focused instructions per agent |
| **Debugging** | Hard to tell which part failed | Easy to isolate issues |
| **Performance** | Sequential execution | Can parallelize searches |
| **Accuracy** | Agent gets confused | Each agent stays on task |
| **Extensibility** | Hard to add features | Easy to add new specialized agents |

## Future Enhancements

Potential improvements:

1. **Email Verification**: Add agent to verify emails are valid
2. **LinkedIn Integration**: Direct LinkedIn API search agent
3. **Caching**: Cache discovered email formats per domain
4. **Parallel Processing**: Run both agents truly in parallel with asyncio.gather
5. **Quality Scoring**: Agent to score confidence of each email guess

## Dependencies

```bash
pip install agents pandas openpyxl
```

Make sure you have OpenAI API key set:
```bash
export OPENAI_API_KEY='your-key-here'
```
