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

### 4. **Retry Logic with Exponential Backoff**

The parallel version includes robust retry handling:
- Automatic retries for transient failures (rate limits, network issues, timeouts)
- Exponential backoff with jitter to avoid thundering herd
- Configurable retry attempts and delays
- Smart detection of retryable vs non-retryable errors
- Default: 3 retries with 2s base delay (2s, 4s, 8s)

### 5. **Structured Outputs**

The parallel version uses strongly-typed structured outputs:
- **EmailFormatAgent** returns `EmailFormatPatterns` TypedDict with validated patterns
- **ExecutiveSearchAgent** returns `ExecutiveList` TypedDict with person objects
- No regex parsing needed - direct JSON extraction
- Guaranteed consistency and correctness
- Better debugging with clear data validation

### 6. **Agent Autonomy**

Agents have flexible search strategies:
- Suggested search approaches rather than rigid instructions
- Full autonomy to choose search order and modify terms
- Can make creative searches based on what they find
- Adapt strategy dynamically during execution

### 7. **Incremental Progress Saving**

Results are saved as they're found:
- CSV file is written after each company is processed
- No data loss if script crashes or is interrupted
- Real-time progress tracking shows contacts found so far
- Can resume by re-running (automatically appends to existing file)

### 8. **More Accurate Results**

Because each agent is specialized:
- Email format searches are more targeted
- Executive searches use better query strategies
- Less confusion about what each step should do
- Clearer output from each stage
- Structured outputs prevent parsing errors

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

## How It Works

### Incremental Saving

The scraper saves results **as each company is processed**, not at the end:

```
📍 PROGRESS: 1/10 companies
🔍 PROCESSING: Acme Corp
✅ Found 2 email format(s)
✅ Found 8 executive(s)
✅ Generated 16 valid emails
   💾 Saved 16 contacts to output.csv
   📊 Total so far: 16 contacts from 1/1 companies

📍 PROGRESS: 2/10 companies
🔍 PROCESSING: TechCo
...
```

**Benefits:**
- If the script crashes on company 5 of 10, you still have results from companies 1-4
- You can monitor the output file in real-time as it grows
- Can stop and resume anytime

### Resuming Interrupted Runs

If you need to resume a run:

```python
# The script will automatically append to existing file
main("companies.csv", "output.csv")

# Output:
# ⚠️  Output file output.csv already exists - will append to it
#    To start fresh, delete the file first
```

To start fresh, delete the output file before running:
```bash
rm output.csv
python multi_agent_email_scraper_parallel.py
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

### Configuring Retry Behavior (Parallel Version)

The parallel version includes configurable retry logic:

```python
from multi_agent_email_scraper_parallel import main

main(
    input_file="companies.csv",
    output_file="contacts.csv",
    max_retries=5,           # Default: 3
    retry_base_delay=3.0     # Default: 2.0 seconds
)
```

**Retry Strategy:**
- **Exponential backoff**: Delays double with each retry (2s → 4s → 8s)
- **Jitter**: Random 10% variation added to prevent thundering herd
- **Max delay**: Capped at 60 seconds
- **Smart detection**: Only retries on rate limits, timeouts, network errors
- **Non-retryable errors**: Fails immediately on logic errors (e.g., max turns exceeded)

**When to adjust:**
- **High rate limits**: Increase `retry_base_delay` to 5.0 or more
- **Unreliable network**: Increase `max_retries` to 5-7
- **Fast API**: Decrease `retry_base_delay` to 1.0

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

### "Generated 0 emails" (despite finding executives and formats)

If you see:
```
✅ Found 1 email format(s)
✅ Found 10 executive(s)
🔧 Generating emails: 10 people, 1 formats
✅ Generated 0 emails
```

**This was fixed in the structured output version!**

**Causes:**
1. Email patterns missing the full domain (e.g., `{first}.{last}` instead of `{first}.{last}@company.com`)
2. Malformed patterns with unreplaced placeholders
3. Name parsing issues (missing first/last names)

**Solutions:**
1. Use the latest version with structured outputs (`output_type=EmailFormatPatterns`)
2. Check debug output - it now shows:
   - `Domain: company.com`
   - `Formats: ['{first}.{last}@company.com']`
   - `⚠️ Skipping malformed email: ...` messages
3. The new version validates patterns include the full `@domain.com`

### Emails with unreplaced placeholders in output

If your CSV contains emails like:
```
{first}.{last}@company.com
j.{last}@company.com
{first}j@company.com
```

**This was fixed with strict pattern validation!**

**What was happening:**
- Agent was returning patterns with invalid placeholders or mixed literals
- Pattern validation wasn't strict enough

**The fix includes:**
1. **Pattern normalization**: Automatically fixes spaces, capitalization, common typos
2. **Strict validation**: Only allows `{first}`, `{last}`, `{f}`, `{l}` placeholders
3. **Clear agent instructions**: Examples of correct and incorrect patterns
4. **Better error messages**: Shows exactly why a pattern was rejected

**Debug output now shows:**
```
   Raw patterns from agent: ['{first}.{last}@company.com', 'j.{last}@company.com']
   ⚠️  Invalid placeholder in pattern: j in j.{last}@company.com
   ⚠️  Rejected invalid pattern: j.{last}@company.com
   Validated patterns: ['{first}.{last}@company.com']
```

**If you still see issues:**
- Check the console output for rejected patterns
- The agent may be finding unusual formats - check the 'notes' field
- Consider adjusting agent instructions if your company uses non-standard formats

### Rate Limit Errors (429)

If you see rate limit errors:
```
⚠️  EmailFormatDiscovery: Attempt 1/3 failed: 429 Rate limit exceeded
   Retrying in 2.3s...
```

**Solutions:**
1. The script will automatically retry with exponential backoff
2. Increase `retry_base_delay` to wait longer between retries
3. Increase `max_retries` if you need more attempts
4. Consider processing fewer companies at once

### Network Timeouts

For timeout errors:
```
⚠️  ExecutiveSearch: Attempt 2/3 failed: Connection timeout
   Retrying in 4.7s...
```

**Solutions:**
1. The script automatically retries network errors
2. Check your internet connection
3. Increase `max_retries` for unreliable networks
4. If persists, check if the web search API is accessible

## Comparison: Single vs Multi-Agent

| Aspect | Single Agent | Multi-Agent | Multi-Agent Parallel |
|--------|--------------|-------------|---------------------|
| **Clarity** | One set of mixed instructions | Clear, focused instructions per agent | Clear, focused instructions per agent |
| **Debugging** | Hard to tell which part failed | Easy to isolate issues | Easy to isolate issues + detailed logging |
| **Performance** | Sequential execution | Can parallelize searches | True parallel execution |
| **Accuracy** | Agent gets confused | Each agent stays on task | Each agent stays on task |
| **Extensibility** | Hard to add features | Easy to add new specialized agents | Easy to add new specialized agents |
| **Reliability** | No retry logic | No retry logic | Exponential backoff retries |
| **Error Handling** | Basic error catching | Basic error catching | Smart retry on transient failures |
| **Output Type** | Free text (regex parsing) | Free text (regex parsing) | Structured TypedDict (validated) |
| **Data Validation** | Prone to parsing errors | Prone to parsing errors | Built-in validation & type checking |
| **Search Strategy** | Rigid search instructions | Rigid search instructions | Flexible, agent has autonomy |
| **Progress Tracking** | None until complete | None until complete | Real-time with incremental saving |
| **Data Loss Risk** | Loses all if crashes | Loses all if crashes | Saves after each company |
| **Resume Capability** | Must restart from beginning | Must restart from beginning | Can resume where left off |

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
