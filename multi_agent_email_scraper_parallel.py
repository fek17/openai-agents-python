"""
Multi-Agent Email Contact Scraper with Parallel Execution

This version runs the EmailFormatAgent and ExecutiveSearchAgent in PARALLEL
using asyncio.gather for maximum performance.

Key improvement: Both agents search simultaneously, reducing total time.
"""

import asyncio
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, TypedDict
import random

import pandas as pd

from agents import Agent, Runner, WebSearchTool, function_tool, trace
from agents.exceptions import MaxTurnsExceeded


# =========================================================
# TypedDict Schemas
# =========================================================

class Person(TypedDict):
    """A senior executive or strategy contact."""
    company_name: str
    first_name: str
    last_name: str
    title: str


class EmailGuess(TypedDict):
    """Structured record for guessed email."""
    company_name: str
    first_name: str
    last_name: str
    title: str
    email_guess: str


# =========================================================
# Helper: Seniority filter
# =========================================================

def is_senior_enough(title: str) -> bool:
    """Check if a job title represents senior leadership."""
    title_lower = title.lower()

    senior_keywords = [
        'ceo', 'cfo', 'cmo', 'coo', 'cto', 'cio', 'cso', 'chro', 'clo', 'cco',
        'chief', 'president', 'chairman', 'chairwoman', 'chair',
        'evp', 'executive vice president',
        'svp', 'senior vice president', 'senior vp',
        'vice president', ' vp ', 'vp of', 'vp,',
        'managing director', 'general manager',
        'director of', 'head of', 'global director', 'regional director',
        'partner', 'managing partner', 'senior partner',
        'board member', 'non-executive director'
    ]

    exclude_keywords = [
        'analyst', 'engineer', 'developer', 'programmer',
        'specialist', 'expert', 'consultant',
        'coordinator', 'administrator', 'associate',
        'product owner', 'scrum master', 'agile coach',
        'architect', 'manager'
    ]

    for keyword in exclude_keywords:
        if keyword in title_lower:
            if keyword in ['analyst', 'engineer', 'developer', 'specialist', 'coordinator']:
                return False

    for keyword in senior_keywords:
        if keyword in title_lower:
            return True

    if ('head of' in title_lower or 'director of' in title_lower or
        'vp of' in title_lower or 'svp of' in title_lower):
        return True

    return False


# =========================================================
# Helper: Character transliteration
# =========================================================

def transliterate_to_ascii(text: str) -> str:
    """Convert accented characters to ASCII for email addresses."""
    if not text:
        return ""

    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
        'ß': 'ss', 'æ': 'ae', 'Æ': 'Ae',
        'œ': 'oe', 'Œ': 'Oe',
        'ø': 'o', 'Ø': 'O',
        'å': 'a', 'Å': 'A',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    nfd = unicodedata.normalize('NFD', text)
    ascii_text = ''.join(
        char for char in nfd
        if unicodedata.category(char) != 'Mn'
    )
    ascii_text = re.sub(r'[^\x00-\x7F]+', '', ascii_text)

    return ascii_text


def sanitize_name(name: str) -> str:
    """Clean and sanitize a name."""
    if not name:
        return ""
    clean = transliterate_to_ascii(name)
    clean = re.sub(r'[^\w\s-]', '', str(clean))
    clean = ' '.join(clean.split())
    return clean.strip()


def get_simple_name(name: str) -> str:
    """Extract simplest version of a name for email."""
    if not name:
        return ""
    name = re.sub(r'[^\w\s-]', '', name)
    if ' ' in name:
        name = name.split()[0]
    if '-' in name:
        name = name.split('-')[0]
    return name.strip()


# =========================================================
# Helper: Duplicate detection
# =========================================================

def detect_duplicates(names: List[Person]) -> List[Person]:
    """Detect and remove duplicate people."""
    if not names:
        return names

    name_groups = {}
    for person in names:
        key = f"{person['first_name'].lower()}_{person['last_name'].lower()}"
        if key not in name_groups:
            name_groups[key] = []
        name_groups[key].append(person)

    cleaned = []
    for key, group in name_groups.items():
        if len(group) > 1:
            titles = [p['title'] for p in group]
            print(f"⚠️  DUPLICATE: {group[0]['first_name']} {group[0]['last_name']} - keeping first: {titles[0]}")
            cleaned.append(group[0])
        else:
            cleaned.append(group[0])

    return cleaned


# =========================================================
# Generate emails (synchronous)
# =========================================================

def generate_emails_sync(
    names: List[Person],
    formats: List[str],
    company_domain: str
) -> List[EmailGuess]:
    """Combine names and formats to generate email guesses."""

    print(f"\n🔧 Generating emails: {len(names)} people, {len(formats)} formats")

    # Use defaults if no formats
    if not formats:
        print(f"   Using default formats")
        formats = [
            f"{{first}}.{{last}}@{company_domain}",
            f"{{first}}{{last}}@{company_domain}",
            f"{{f}}{{last}}@{company_domain}",
        ]

    # Check duplicates
    names = detect_duplicates(names)

    results = []

    for n in names:
        company, first, last, title = n["company_name"], n["first_name"], n["last_name"], n["title"]

        if not first or not last:
            continue

        # Sanitize names
        first_clean = sanitize_name(first)
        last_clean = sanitize_name(last)

        if not first_clean or not last_clean:
            continue

        first_simple = get_simple_name(first_clean)
        last_simple = get_simple_name(last_clean)

        for fmt in formats:
            replacements = [
                ("{firstname}", first_simple.lower()),
                ("{lastname}", last_simple.lower()),
                ("{first}", first_simple.lower()),
                ("{last}", last_simple.lower()),
                ("{f}", first_simple[0].lower() if first_simple else ""),
                ("{l}", last_simple[0].lower() if last_simple else ""),
            ]

            email = fmt
            for placeholder, value in replacements:
                email = email.replace(placeholder, value)

            if not email.endswith(company_domain):
                if '@' not in email:
                    email = f"{email}@{company_domain}"

            if '{' in email or '}' in email:
                continue

            if not re.match(r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                continue

            results.append({
                "company_name": company,
                "first_name": first_clean,
                "last_name": last_clean,
                "title": title,
                "email_guess": email,
            })

    print(f"✅ Generated {len(results)} emails\n")
    return results


# =========================================================
# SPECIALIZED AGENTS
# =========================================================

email_format_agent = Agent(
    name="EmailFormatDiscovery",
    instructions=(
        "You are an expert at discovering email format patterns for companies.\n\n"

        "Your ONLY job is to find the email format(s) used by the given company.\n\n"

        "Search strategies:\n"
        "1. Search for: '{company_name} email format'\n"
        "2. Search for: '{company_name} contact email address'\n"
        "3. Search for: 'site:{company_domain} contact email'\n"
        "4. Look for patterns like:\n"
        "   - firstname.lastname@domain.com\n"
        "   - firstnamelastname@domain.com\n"
        "   - first.last@domain.com\n"
        "   - f.lastname@domain.com\n"
        "   - flastname@domain.com\n\n"

        "Output format:\n"
        "Return a list of format patterns using placeholders:\n"
        "- {first} or {firstname} for first name\n"
        "- {last} or {lastname} for last name\n"
        "- {f} for first initial\n"
        "- {l} for last initial\n\n"

        "Examples of formats to return:\n"
        "- '{first}.{last}@company.com'\n"
        "- '{first}{last}@company.com'\n"
        "- '{f}{last}@company.com'\n\n"

        "Return 1-3 most likely patterns based on your search.\n"
        "If you cannot find any patterns, say 'No specific patterns found'.\n"
    ),
    tools=[WebSearchTool()],
    model="gpt-4o-mini",
)


executive_search_agent = Agent(
    name="ExecutiveSearch",
    instructions=(
        "You are an expert at finding senior executives at companies.\n\n"

        "Your ONLY job is to find C-suite executives, VPs, and Directors.\n\n"

        "Search strategies:\n"
        "1. Search for: '{company_name} CEO CFO CTO executives'\n"
        "2. Search for: '{company_name} leadership team'\n"
        "3. Search for: '{company_name} senior management'\n"
        "4. Search for: 'site:{company_domain} about team leadership'\n"
        "5. Search LinkedIn: '{company_name} site:linkedin.com CEO CFO CTO'\n\n"

        "Target roles (ONLY include these):\n"
        "- C-suite: CEO, CFO, CTO, CMO, COO, CIO, CSO, CHRO, etc.\n"
        "- EVP / Executive Vice President\n"
        "- SVP / Senior Vice President\n"
        "- VP / Vice President\n"
        "- Managing Director\n"
        "- Director of [major department]\n"
        "- Head of [major department]\n"
        "- Partner / Managing Partner\n\n"

        "DO NOT include:\n"
        "- Managers, Coordinators, Specialists\n"
        "- Engineers, Developers, Analysts\n"
        "- Consultants, Associates\n\n"

        "For each person found, provide in this EXACT format:\n"
        "Name: [First Last]\n"
        "Title: [Job Title]\n"
        "---\n\n"

        "Aim to find 5-15 senior executives per company.\n"
    ),
    tools=[WebSearchTool()],
    model="gpt-4o-mini",
)


# =========================================================
# Parse agent outputs
# =========================================================

def parse_email_formats(output: str) -> List[str]:
    """Parse email format patterns from agent output."""
    if "No specific patterns found" in output:
        return []

    # Look for format patterns in the output
    patterns = []
    pattern_regex = r'\{(?:first|firstname|last|lastname|f|l)\}[.\-_]?\{?(?:first|firstname|last|lastname|f|l)?\}?@[\w\.-]+'

    found_patterns = re.findall(pattern_regex, output)
    if found_patterns:
        patterns.extend(found_patterns)

    # Also look for common patterns mentioned in text
    common_patterns = [
        r'\{first\}\.\{last\}',
        r'\{first\}\{last\}',
        r'\{f\}\{last\}',
        r'\{firstname\}\.\{lastname\}',
        r'\{f\}\.\{last\}',
    ]

    for pattern in common_patterns:
        if pattern in output:
            patterns.append(pattern)

    return list(set(patterns))  # Remove duplicates


def parse_executives(output: str, company_name: str) -> List[Person]:
    """Parse executive names and titles from agent output."""
    executives = []

    # Split by --- or double newline
    entries = re.split(r'---+|\n\n', output)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Look for Name: and Title: patterns
        name_match = re.search(r'Name:\s*([^\n]+)', entry, re.IGNORECASE)
        title_match = re.search(r'Title:\s*([^\n]+)', entry, re.IGNORECASE)

        if name_match and title_match:
            full_name = name_match.group(1).strip()
            title = title_match.group(1).strip()

            # Split name into first and last
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = ' '.join(name_parts[1:])

                executives.append({
                    'company_name': company_name,
                    'first_name': first_name,
                    'last_name': last_name,
                    'title': title
                })

    return executives


# =========================================================
# Retry logic with exponential backoff
# =========================================================

async def run_agent_with_retry(
    agent: Agent,
    prompt: str,
    max_turns: int = 20,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0
):
    """
    Run an agent with exponential backoff retry logic.

    Args:
        agent: The agent to run
        prompt: The prompt to send to the agent
        max_turns: Maximum turns for the agent
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)
        max_delay: Maximum delay in seconds (default: 60.0)

    Returns:
        Agent result or raises exception after all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            result = await Runner.run(agent, prompt, max_turns=max_turns)
            return result

        except MaxTurnsExceeded as e:
            # Don't retry if max turns exceeded - this is a logic issue, not transient
            print(f"⚠️  {agent.name}: Max turns exceeded (not retrying)")
            raise e

        except Exception as e:
            last_exception = e

            # Check if this is a rate limit or network error (retryable)
            error_msg = str(e).lower()
            is_retryable = any(keyword in error_msg for keyword in [
                'rate limit',
                'timeout',
                'connection',
                'network',
                'temporarily unavailable',
                'service unavailable',
                '429',  # HTTP 429 Too Many Requests
                '500',  # HTTP 500 Internal Server Error
                '502',  # HTTP 502 Bad Gateway
                '503',  # HTTP 503 Service Unavailable
                '504',  # HTTP 504 Gateway Timeout
            ])

            if not is_retryable:
                print(f"❌ {agent.name}: Non-retryable error: {e}")
                raise e

            if attempt < max_retries - 1:
                # Calculate exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
                total_delay = delay + jitter

                print(f"⚠️  {agent.name}: Attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"   Retrying in {total_delay:.1f}s...")

                await asyncio.sleep(total_delay)
            else:
                print(f"❌ {agent.name}: All {max_retries} retry attempts exhausted")
                raise last_exception

    # Should never reach here, but just in case
    raise last_exception


# =========================================================
# Main parallel processing function
# =========================================================

async def process_company_parallel(
    company_name: str,
    company_domain: str,
    company_website: str = "",
    output_file: str = "contacts.csv",
    max_turns: int = 20,
    max_retries: int = 3,
    retry_base_delay: float = 2.0
) -> tuple[pd.DataFrame | None, str]:
    """
    Process a single company using parallel agent execution with retry logic.

    Both agents run simultaneously for better performance.

    Args:
        company_name: Name of the company
        company_domain: Company domain (e.g., 'company.com')
        company_website: Full website URL (optional)
        output_file: Output CSV file path
        max_turns: Maximum turns per agent
        max_retries: Maximum retry attempts for transient failures (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 2.0)

    Returns:
        Tuple of (DataFrame with results or None, output file path)
    """

    print(f"\n{'='*60}")
    print(f"🔍 PROCESSING: {company_name}")
    print(f"{'='*60}\n")

    website_info = f"\nWebsite: {company_website}" if company_website else ""

    # Prepare prompts for both agents
    format_prompt = f"""
Find the email format for {company_name}.{website_info}
Domain: {company_domain}

Search for their email format and return the pattern(s).
"""

    executive_prompt = f"""
Find senior executives at {company_name}.{website_info}
Domain: {company_domain}

Search for C-suite, VPs, and Directors. Return in the format:
Name: [First Last]
Title: [Job Title]
---
"""

    try:
        # Run both agents IN PARALLEL using asyncio.gather with retry logic
        with trace(f"Processing {company_name}"):
            print("🚀 Running both agents in parallel with retry support...")

            format_result, executive_result = await asyncio.gather(
                run_agent_with_retry(
                    email_format_agent,
                    format_prompt,
                    max_turns=max_turns,
                    max_retries=max_retries,
                    base_delay=retry_base_delay
                ),
                run_agent_with_retry(
                    executive_search_agent,
                    executive_prompt,
                    max_turns=max_turns,
                    max_retries=max_retries,
                    base_delay=retry_base_delay
                ),
                return_exceptions=True
            )

            # Check for exceptions
            if isinstance(format_result, Exception):
                print(f"⚠️  Email format agent failed: {format_result}")
                email_formats = []
            else:
                email_formats = parse_email_formats(str(format_result.final_output))
                print(f"✅ Found {len(email_formats)} email format(s)")

            if isinstance(executive_result, Exception):
                print(f"⚠️  Executive search agent failed: {executive_result}")
                executives = []
            else:
                executives = parse_executives(str(executive_result.final_output), company_name)
                print(f"✅ Found {len(executives)} executive(s)")

            # Generate emails
            if executives:
                email_guesses = generate_emails_sync(
                    names=executives,
                    formats=email_formats,
                    company_domain=company_domain
                )

                if email_guesses:
                    # Export to CSV
                    df = pd.DataFrame(email_guesses)
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"✅ Exported {len(email_guesses)} contacts to {output_file}")
                    return df, output_file
                else:
                    print("⚠️  No emails generated")
                    return None, output_file
            else:
                print("⚠️  No executives found")
                return None, output_file

    except MaxTurnsExceeded:
        print(f"⚠️  Max turns exceeded")
        return None, output_file
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, output_file


# =========================================================
# Batch processing
# =========================================================

async def process_multiple_companies(
    companies: List[tuple[str, str, str]],
    output_dir: str = ".",
    max_retries: int = 3,
    retry_base_delay: float = 2.0
) -> List[pd.DataFrame]:
    """
    Process multiple companies in parallel with retry logic.

    Args:
        companies: List of (company_name, domain, website) tuples
        output_dir: Directory to save output files
        max_retries: Maximum retry attempts for transient failures (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 2.0)

    Returns:
        List of DataFrames with results
    """
    print(f"\n🚀 Processing {len(companies)} companies in parallel...\n")

    tasks = []
    for company_name, domain, website in companies:
        safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
        output_file = f"{output_dir}/{safe_name}_contacts.csv"

        tasks.append(
            process_company_parallel(
                company_name=company_name,
                company_domain=domain,
                company_website=website,
                output_file=output_file,
                max_turns=20,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay
            )
        )

    # Run all companies in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results
    successful_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"⚠️  Company processing failed: {result}")
        else:
            df, _ = result
            if df is not None:
                successful_results.append(df)

    return successful_results


# =========================================================
# Main execution
# =========================================================

def main(
    input_file: str,
    output_file: str = "all_contacts_output.csv",
    max_retries: int = 3,
    retry_base_delay: float = 2.0
):
    """
    Process multiple companies from spreadsheet with retry logic.

    Args:
        input_file: Path to CSV or Excel file with companies
        output_file: Path for output CSV file
        max_retries: Maximum retry attempts for transient failures (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 2.0)
    """

    print(f"📂 Reading: {input_file}")

    file_path = Path(input_file)
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                break
            except:
                continue
        if df is None:
            raise Exception("Could not read file")

    print(f"✅ Loaded {len(df)} companies\n")

    company_col = df.columns[0]
    website_col = df.columns[1] if len(df.columns) > 1 else None

    # Prepare company list
    companies = []
    for idx, row in df.iterrows():
        company_name = row[company_col]

        if pd.isna(company_name) or str(company_name).strip() == "":
            continue

        company_name = str(company_name).strip()
        website = str(row[website_col]).strip() if website_col and not pd.isna(row[website_col]) else ""

        # Extract domain
        company_domain = ""
        if website:
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', website)
            if domain_match:
                company_domain = domain_match.group(1)

        if not company_domain:
            company_domain = company_name.lower().replace(' ', '').replace(',', '').replace('.', '') + '.com'

        companies.append((company_name, company_domain, website))

    # Process all companies
    all_results = []

    async def process_all():
        for company_name, company_domain, website in companies:
            safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
            temp_csv = f"temp_{safe_name}_contacts.csv"

            result_df, _ = await process_company_parallel(
                company_name=company_name,
                company_domain=company_domain,
                company_website=website,
                output_file=temp_csv,
                max_turns=20,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay
            )

            if result_df is not None and len(result_df) > 0:
                for _, contact_row in result_df.iterrows():
                    title = contact_row.get("title", "")

                    if is_senior_enough(title):
                        all_results.append({
                            "Company Name": company_name,
                            "First Name": contact_row.get("first_name", ""),
                            "Last Name": contact_row.get("last_name", ""),
                            "Job Title": title,
                            "Email Guess": contact_row.get("email_guess", "")
                        })

                # Clean up temp file
                if Path(temp_csv).exists():
                    Path(temp_csv).unlink()

    # Run async processing
    asyncio.run(process_all())

    # Export final results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n{'='*60}")
        print(f"✅ DONE! {len(all_results)} contacts → {output_file}")
        print(f"{'='*60}")
    else:
        print("\n⚠️  No results")


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    # Update these paths for your environment
    INPUT_FILE = "input_companies.csv"  # or .xlsx
    OUTPUT_FILE = "output_contacts.csv"

    main(INPUT_FILE, OUTPUT_FILE)
