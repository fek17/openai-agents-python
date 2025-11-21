"""
Multi-Agent Email Contact Scraper

This script uses a multi-agent architecture to scrape senior executive contacts:
- EmailFormatAgent: Discovers email format patterns for a company
- ExecutiveSearchAgent: Finds senior executives at a company
- CoordinatorAgent: Orchestrates both agents and combines results

The parallel execution improves speed and accuracy.
"""

import asyncio
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, TypedDict

import pandas as pd

from agents import Agent, Runner, WebSearchTool, function_tool
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


class EmailFormat(TypedDict):
    """Email format pattern discovered for a company."""
    pattern: str
    confidence: str
    example: str


class EmailFormatResult(TypedDict):
    """Result from email format discovery."""
    company_name: str
    domain: str
    formats: List[str]
    notes: str


class ExecutiveSearchResult(TypedDict):
    """Result from executive search."""
    company_name: str
    executives: List[Person]
    count: int


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
# Generate emails
# =========================================================

@function_tool
def generate_emails(
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
# Export to CSV
# =========================================================

@function_tool
def export_to_csv(data: List[EmailGuess], filename: str = "contacts.csv") -> str:
    """Export contact data to CSV."""
    if not data:
        return "No data to export."

    print(f"💾 Exporting {len(data)} rows to {filename}")

    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Exported\n")
    return f"Exported {len(data)} rows"


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

        "Examples:\n"
        "- '{first}.{last}@company.com'\n"
        "- '{first}{last}@company.com'\n"
        "- '{f}{last}@company.com'\n\n"

        "Return 1-3 most likely patterns based on your search.\n"
        "If you cannot find any patterns, return an empty list [].\n"
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

        "For each person found, provide:\n"
        "- company_name: The company name\n"
        "- first_name: First name only\n"
        "- last_name: Last name only\n"
        "- title: Their exact job title\n\n"

        "Aim to find 5-15 senior executives per company.\n"
    ),
    tools=[WebSearchTool()],
    model="gpt-4o-mini",
)


# =========================================================
# Custom output extractors
# =========================================================

async def _extract_email_formats(run_result) -> List[str]:
    """Extract email format patterns from the EmailFormatAgent result."""
    output = run_result.final_output

    # If output is already a list, return it
    if isinstance(output, list):
        return output

    # Try to parse the output as a list of format strings
    output_str = str(output)

    # Look for format patterns in the output
    patterns = []
    pattern_regex = r'\{(?:first|firstname|last|lastname|f|l)\}[.\-_]?\{?(?:first|firstname|last|lastname|f|l)?\}?@[\w\.-]+'

    found_patterns = re.findall(pattern_regex, output_str)
    if found_patterns:
        patterns.extend(found_patterns)

    # If no patterns found, return empty list
    return patterns if patterns else []


async def _extract_executives(run_result) -> List[Person]:
    """Extract executive list from the ExecutiveSearchAgent result."""
    output = run_result.final_output

    # If output is already a list, return it
    if isinstance(output, list):
        return output

    # Otherwise return empty list (the agent should handle structuring)
    return []


# =========================================================
# COORDINATOR AGENT
# =========================================================

coordinator_agent = Agent(
    name="ContactCoordinator",
    instructions=(
        "You coordinate the email contact scraping process.\n\n"

        "You have access to two specialized agents:\n"
        "1. discover_email_formats: Finds email format patterns for the company\n"
        "2. find_executives: Finds senior executives at the company\n\n"

        "Your workflow:\n"
        "1. Call BOTH agents (you can call them in parallel or sequentially)\n"
        "2. Once you have both results, call generate_emails with:\n"
        "   - names: the list of executives\n"
        "   - formats: the list of email format patterns\n"
        "   - company_domain: the domain\n"
        "3. Call export_to_csv with the generated emails and the output filename\n\n"

        "IMPORTANT:\n"
        "- You MUST call all 4 tools: discover_email_formats, find_executives, generate_emails, export_to_csv\n"
        "- If either agent returns empty results, still proceed with defaults\n"
        "- Always complete all steps\n"
    ),
    tools=[
        email_format_agent.as_tool(
            tool_name="discover_email_formats",
            tool_description="Discovers email format patterns for the company. Returns list of format strings.",
            custom_output_extractor=_extract_email_formats,
        ),
        executive_search_agent.as_tool(
            tool_name="find_executives",
            tool_description="Finds senior executives at the company. Returns list of Person objects with first_name, last_name, title, company_name.",
            custom_output_extractor=_extract_executives,
        ),
        generate_emails,
        export_to_csv,
    ],
    model="gpt-4o-mini",
)


# =========================================================
# Process single company (async version)
# =========================================================

async def process_company_async(
    company_name: str,
    company_domain: str,
    company_website: str = "",
    temp_file_prefix: str = "temp",
    max_turns: int = 30
):
    """Process a single company using multi-agent system."""

    print(f"\n{'='*60}")
    print(f"🔍 PROCESSING: {company_name}")
    print(f"{'='*60}\n")

    safe_company_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
    temp_csv = f"{temp_file_prefix}_{safe_company_name}_contacts.csv"

    website_info = f"\nWebsite: {company_website}" if company_website else ""

    prompt = f"""
Find senior executives for {company_name} and generate their emails.{website_info}
Domain: {company_domain}
Output file: {temp_csv}

Steps:
1. Use discover_email_formats to find email patterns
2. Use find_executives to find senior executives
3. Call generate_emails with the results
4. Call export_to_csv to save the results

Complete all 4 steps.
"""

    try:
        result = await Runner.run(coordinator_agent, prompt, max_turns=max_turns)

        if Path(temp_csv).exists():
            print(f"✅ CSV created: {temp_csv}")
            return pd.read_csv(temp_csv), temp_csv
        else:
            print(f"⚠️  No CSV created")
            return None, temp_csv

    except MaxTurnsExceeded:
        print(f"⚠️  Max turns exceeded")
        return None, temp_csv
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, temp_csv


# =========================================================
# Synchronous wrapper
# =========================================================

def process_company(
    company_name: str,
    company_domain: str,
    company_website: str = "",
    temp_file_prefix: str = "temp",
    max_turns: int = 30
):
    """Synchronous wrapper for process_company_async."""
    return asyncio.run(
        process_company_async(
            company_name,
            company_domain,
            company_website,
            temp_file_prefix,
            max_turns
        )
    )


# =========================================================
# Main execution
# =========================================================

def main(input_file: str, output_file: str = "all_contacts_output.csv"):
    """Process multiple companies from spreadsheet."""

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

    all_results = []

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

        print(f"{'='*60}")
        print(f"🔍 {idx + 1}/{len(df)}: {company_name}")
        print(f"   Domain: {company_domain}")
        if website:
            print(f"   Website: {website}")
        print(f"{'='*60}")

        try:
            result_df, temp_csv = process_company(
                company_name=company_name,
                company_domain=company_domain,
                company_website=website,
                temp_file_prefix="temp_company",
                max_turns=30
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

                print(f"✅ Added {len(result_df)} contacts")

                if Path(temp_csv).exists():
                    Path(temp_csv).unlink()

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

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
