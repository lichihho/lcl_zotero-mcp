"""
Zotero client wrapper for MCP server.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from markitdown import MarkItDown
from pyzotero import zotero

from zotero_mcp.utils import format_creators

# Load environment variables
load_dotenv()


@dataclass
class AttachmentDetails:
    """Details about a Zotero attachment."""

    key: str
    title: str
    filename: str
    content_type: str


def get_zotero_client() -> zotero.Zotero:
    """
    Get authenticated Zotero client using environment variables.

    Returns:
        A configured Zotero client instance.

    Raises:
        ValueError: If required environment variables are missing.
    """
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    api_key = os.getenv("ZOTERO_API_KEY")
    local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]

    # For local API, default to user ID 0 if not specified
    if local and not library_id:
        library_id = "0"

    # For remote API, we need both library_id and api_key
    if not local and not (library_id and api_key):
        raise ValueError(
            "Missing required environment variables. Please set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY, "
            "or use ZOTERO_LOCAL=true for local Zotero instance."
        )

    return zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        api_key=api_key,
        local=local,
    )


def format_item_metadata(item: dict[str, Any], include_abstract: bool = True) -> str:
    """
    Format a Zotero item's metadata as markdown.

    Args:
        item: A Zotero item dictionary.
        include_abstract: Whether to include the abstract in the output.

    Returns:
        Markdown-formatted metadata.
    """
    data = item.get("data", {})
    item_type = data.get("itemType", "unknown")

    # Basic information
    lines = [
        f"# {data.get('title', 'Untitled')}",
        f"**Type:** {item_type}",
        f"**Item Key:** {data.get('key')}",
    ]

    # Date
    if date := data.get("date"):
        lines.append(f"**Date:** {date}")

    # Authors/Creators
    if creators := data.get("creators", []):
        lines.append(f"**Authors:** {format_creators(creators)}")

    # Publication details based on item type
    if item_type == "journalArticle":
        if journal := data.get("publicationTitle"):
            journal_info = f"**Journal:** {journal}"
            if volume := data.get("volume"):
                journal_info += f", Volume {volume}"
            if issue := data.get("issue"):
                journal_info += f", Issue {issue}"
            if pages := data.get("pages"):
                journal_info += f", Pages {pages}"
            lines.append(journal_info)
    elif item_type == "book":
        if publisher := data.get("publisher"):
            book_info = f"**Publisher:** {publisher}"
            if place := data.get("place"):
                book_info += f", {place}"
            lines.append(book_info)

    # DOI and URL
    if doi := data.get("DOI"):
        lines.append(f"**DOI:** {doi}")
    if url := data.get("url"):
        lines.append(f"**URL:** {url}")

    # Extra field often holds citation key / misc metadata
    if extra := data.get("extra"):
        lines.extend(["", "## Extra", extra])

        # Try to surface a citation key if present in Extra
        for line in extra.splitlines():
            if "citation key" in line.lower():
                key_part = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                lines.append(f"**Citation Key (from Extra):** {key_part}")
                break
    
    # Tags
    if tags := data.get("tags"):
        tag_list = [f"`{tag['tag']}`" for tag in tags]
        if tag_list:
            lines.append(f"**Tags:** {' '.join(tag_list)}")

    # Abstract
    if include_abstract and (abstract := data.get("abstractNote")):
        lines.extend(["", "## Abstract", abstract])

    # Collections
    if collections := data.get("collections", []):
        if collections:
            lines.append(f"**Collections:** {len(collections)} collections")

    # Notes - this requires additional API calls, so we just indicate if there are notes
    if "meta" in item and item["meta"].get("numChildren", 0) > 0:
        lines.append(f"**Notes/Attachments:** {item['meta']['numChildren']}")

    return "\n\n".join(lines)


def generate_bibtex(item: dict[str, Any]) -> str:
    """
    Generate BibTeX format for a Zotero item.

    Args:
        item: Zotero item data

    Returns:
        BibTeX formatted string
    """
    data = item.get("data", {})
    item_key = data.get("key")

    # Try Better BibTeX first
    try:
        from zotero_mcp.better_bibtex_client import ZoteroBetterBibTexAPI
        bibtex = ZoteroBetterBibTexAPI()

        if bibtex.is_zotero_running():
            return bibtex.export_bibtex(item_key)

    except Exception:
        # Continue to fallback method if Better BibTeX fails
        pass

    # Fallback to basic BibTeX generation
    item_type = data.get("itemType", "misc")

    if item_type in ["attachment", "note"]:
        raise ValueError(f"Cannot export BibTeX for item type '{item_type}'")

    # Map Zotero item types to BibTeX types
    type_map = {
        "journalArticle": "article",
        "book": "book",
        "bookSection": "incollection",
        "conferencePaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
        "webpage": "misc",
        "manuscript": "unpublished"
    }

    # Create citation key
    creators = data.get("creators", [])
    author = ""
    if creators:
        first = creators[0]
        author = first.get("lastName", first.get("name", "").split()[-1] if first.get("name") else "").replace(" ", "")

    year = data.get("date", "")[:4] if data.get("date") else "nodate"
    cite_key = f"{author}{year}_{item_key}"

    # Build BibTeX entry
    bib_type = type_map.get(item_type, "misc")
    lines = [f"@{bib_type}{{{cite_key},"]

    # Add fields
    field_mappings = [
        ("title", "title"),
        ("publicationTitle", "journal"),
        ("volume", "volume"),
        ("issue", "number"),
        ("pages", "pages"),
        ("publisher", "publisher"),
        ("DOI", "doi"),
        ("url", "url"),
        ("abstractNote", "abstract")
    ]

    for zotero_field, bibtex_field in field_mappings:
        if value := data.get(zotero_field):
            # Escape special characters
            value = value.replace("{", "\\{").replace("}", "\\}")
            lines.append(f'  {bibtex_field} = {{{value}}},')

    # Add authors
    if creators:
        authors = []
        for creator in creators:
            if creator.get("creatorType") == "author":
                if "lastName" in creator and "firstName" in creator:
                    authors.append(f"{creator['lastName']}, {creator['firstName']}")
                elif "name" in creator:
                    authors.append(creator["name"])
        if authors:
            lines.append(f'  author = {{{" and ".join(authors)}}},')

    # Add year
    if year != "nodate":
        lines.append(f'  year = {{{year}}},')

    # Remove trailing comma from last field and close entry
    if lines[-1].endswith(','):
        lines[-1] = lines[-1][:-1]
    lines.append("}")

    return "\n".join(lines)


def _parse_bibtex_authors(author_string: str) -> list[dict[str, str]]:
    """
    Parse BibTeX author string into Zotero creator format.

    Handles:
    - "Last, First and Last, First" format
    - "First Last and First Last" format

    Args:
        author_string: BibTeX-formatted author string.

    Returns:
        List of Zotero creator dicts.
    """
    creators = []
    authors = [a.strip() for a in author_string.split(" and ")]
    for author in authors:
        if not author:
            continue
        if author.lower() == "others":
            continue
        if "," in author:
            parts = author.split(",", 1)
            last_name = parts[0].strip()
            first_name = parts[1].strip()
        else:
            parts = author.rsplit(" ", 1)
            if len(parts) == 2:
                first_name = parts[0].strip()
                last_name = parts[1].strip()
            else:
                first_name = ""
                last_name = parts[0].strip()
        creators.append({
            "creatorType": "author",
            "firstName": first_name,
            "lastName": last_name,
        })
    return creators


def _strip_latex(text: str) -> str:
    """
    Remove common LaTeX escape sequences and convert to Unicode.

    Args:
        text: String possibly containing LaTeX commands.

    Returns:
        Cleaned string with Unicode characters.
    """
    import re

    if not text:
        return text

    result = text

    replacements = [
        # Umlaut (diaeresis)
        ('{\\"a}', '\u00e4'), ('{\\"o}', '\u00f6'), ('{\\"u}', '\u00fc'),
        ('{\\"A}', '\u00c4'), ('{\\"O}', '\u00d6'), ('{\\"U}', '\u00dc'),
        ('{\\"e}', '\u00eb'), ('{\\"i}', '\u00ef'),
        ('\\"a', '\u00e4'), ('\\"o', '\u00f6'), ('\\"u', '\u00fc'),
        ('\\"A', '\u00c4'), ('\\"O', '\u00d6'), ('\\"U', '\u00dc'),
        # Acute
        ("{\\'a}", '\u00e1'), ("{\\'e}", '\u00e9'), ("{\\'i}", '\u00ed'),
        ("{\\'o}", '\u00f3'), ("{\\'u}", '\u00fa'),
        ("{\\'A}", '\u00c1'), ("{\\'E}", '\u00c9'), ("{\\'O}", '\u00d3'),
        ("{\\'U}", '\u00da'),
        ("\\'a", '\u00e1'), ("\\'e", '\u00e9'), ("\\'i", '\u00ed'),
        ("\\'o", '\u00f3'), ("\\'u", '\u00fa'),
        # Grave
        ('{\\`a}', '\u00e0'), ('{\\`e}', '\u00e8'), ('{\\`o}', '\u00f2'),
        ('{\\`u}', '\u00f9'),
        ('\\`a', '\u00e0'), ('\\`e', '\u00e8'), ('\\`o', '\u00f2'),
        ('\\`u', '\u00f9'),
        # Circumflex
        ('{\\^a}', '\u00e2'), ('{\\^e}', '\u00ea'), ('{\\^o}', '\u00f4'),
        ('{\\^u}', '\u00fb'),
        ('\\^a', '\u00e2'), ('\\^e', '\u00ea'), ('\\^o', '\u00f4'),
        ('\\^u', '\u00fb'),
        # Tilde
        ('{\\~a}', '\u00e3'), ('{\\~n}', '\u00f1'), ('{\\~o}', '\u00f5'),
        ('\\~a', '\u00e3'), ('\\~n', '\u00f1'), ('\\~o', '\u00f5'),
        # Caron
        ('{\\v{c}}', '\u010d'), ('{\\v{s}}', '\u0161'), ('{\\v{z}}', '\u017e'),
        ('{\\v c}', '\u010d'), ('{\\v s}', '\u0161'), ('{\\v z}', '\u017e'),
        ('\\v{c}', '\u010d'), ('\\v{s}', '\u0161'), ('\\v{z}', '\u017e'),
        ('{\\v{C}}', '\u010c'), ('{\\v{S}}', '\u0160'), ('{\\v{Z}}', '\u017d'),
        ('\\v{C}', '\u010c'), ('\\v{S}', '\u0160'), ('\\v{Z}', '\u017d'),
        # Cedilla
        ('{\\c{c}}', '\u00e7'), ('{\\c c}', '\u00e7'), ('\\c{c}', '\u00e7'),
        ('{\\c{C}}', '\u00c7'), ('\\c{C}', '\u00c7'),
        # Misc
        ('{\\ss}', '\u00df'), ('\\ss', '\u00df'),
        ('{\\ae}', '\u00e6'), ('\\ae', '\u00e6'),
        ('{\\oe}', '\u0153'), ('\\oe', '\u0153'),
        ('{\\AE}', '\u00c6'), ('\\AE', '\u00c6'),
        ('{\\OE}', '\u0152'), ('\\OE', '\u0152'),
        ('{\\aa}', '\u00e5'), ('\\aa', '\u00e5'),
        ('{\\AA}', '\u00c5'), ('\\AA', '\u00c5'),
        ('{\\o}', '\u00f8'), ('\\o', '\u00f8'),
        ('{\\O}', '\u00d8'), ('\\O', '\u00d8'),
    ]

    for latex, unicode_char in replacements:
        result = result.replace(latex, unicode_char)

    result = result.replace("{", "").replace("}", "")
    result = re.sub(r'\\(?:textit|textbf|textrm|textsf|texttt|emph)\s*', '', result)

    return result.strip()


def parse_bibtex_to_zotero_items(
    bibtex_string: str,
    zot: Any,
    collection_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Parse BibTeX string and convert to Zotero item dicts.

    Args:
        bibtex_string: BibTeX-formatted string containing one or more entries.
        zot: Authenticated pyzotero client (used for item_template).
        collection_key: Optional collection key to assign items to.

    Returns:
        List of Zotero item dicts ready for create_items().
    """
    import bibtexparser

    bib_db = bibtexparser.loads(bibtex_string)

    type_map = {
        "article": "journalArticle",
        "inproceedings": "conferencePaper",
        "conference": "conferencePaper",
        "book": "book",
        "incollection": "bookSection",
        "phdthesis": "thesis",
        "mastersthesis": "thesis",
        "techreport": "report",
        "misc": "document",
        "unpublished": "manuscript",
        "inbook": "bookSection",
        "proceedings": "book",
    }

    field_map = {
        "title": "title",
        "journal": "publicationTitle",
        "volume": "volume",
        "number": "issue",
        "pages": "pages",
        "year": "date",
        "doi": "DOI",
        "url": "url",
        "abstract": "abstractNote",
        "publisher": "publisher",
        "address": "place",
        "isbn": "ISBN",
        "issn": "ISSN",
        "note": "extra",
        "keywords": "tags",
    }

    items = []
    for entry in bib_db.entries:
        entry_type = entry.get("ENTRYTYPE", "misc").lower()
        item_type = type_map.get(entry_type, "document")
        template = zot.item_template(item_type)

        for bib_field, zotero_field in field_map.items():
            value = entry.get(bib_field)
            if not value:
                continue
            raw = _strip_latex(value)

            if zotero_field == "tags":
                keywords = [k.strip() for k in raw.split(",") if k.strip()]
                template["tags"] = [{"tag": kw} for kw in keywords]
            else:
                template[zotero_field] = raw

        booktitle = entry.get("booktitle")
        if booktitle:
            bt_value = _strip_latex(booktitle)
            if item_type == "conferencePaper":
                template["proceedingsTitle"] = bt_value
            elif item_type == "bookSection":
                template["bookTitle"] = bt_value

        author = entry.get("author")
        if author:
            template["creators"] = _parse_bibtex_authors(_strip_latex(author))

        editor = entry.get("editor")
        if editor:
            editors = _parse_bibtex_authors(_strip_latex(editor))
            for ed in editors:
                ed["creatorType"] = "editor"
            existing = template.get("creators", [])
            template["creators"] = existing + editors

        if collection_key:
            template["collections"] = [collection_key]

        items.append(template)

    return items


def get_attachment_details(
    zot: zotero.Zotero, item: dict[str, Any]
) -> AttachmentDetails | None:
    """
    Get attachment details for a Zotero item, finding the most relevant attachment.

    Args:
        zot: A Zotero client instance.
        item: A Zotero item dictionary.

    Returns:
        AttachmentDetails if found, None otherwise.
    """
    data = item.get("data", {})
    item_type = data.get("itemType")
    item_key = data.get("key")

    # Direct attachment
    if item_type == "attachment":
        return AttachmentDetails(
            key=item_key,
            title=data.get("title", "Untitled"),
            filename=data.get("filename", ""),
            content_type=data.get("contentType", ""),
        )

    # For regular items, look for child attachments
    try:
        children = zot.children(item_key)

        # Group attachments by content type
        pdfs = []
        htmls = []
        others = []

        for child in children:
            child_data = child.get("data", {})
            if child_data.get("itemType") == "attachment":
                content_type = child_data.get("contentType", "")
                filename = child_data.get("filename", "")
                title = child_data.get("title", "Untitled")
                key = child.get("key", "")

                # Use MD5 as proxy for size (longer MD5 usually means larger file)
                size_proxy = len(child_data.get("md5", ""))

                attachment = (key, title, filename, content_type, size_proxy)

                if content_type == "application/pdf":
                    pdfs.append(attachment)
                elif content_type.startswith("text/html"):
                    htmls.append(attachment)
                else:
                    others.append(attachment)

        # Return first match in priority order (PDF > HTML > other)
        # Sort each category by size (descending) to get largest/most complete file
        for category in [pdfs, htmls, others]:
            if category:
                category.sort(key=lambda x: x[4], reverse=True)
                key, title, filename, content_type, _ = category[0]
                return AttachmentDetails(
                    key=key,
                    title=title,
                    filename=filename,
                    content_type=content_type,
                )
    except Exception:
        pass

    return None


def convert_to_markdown(file_path: str | Path) -> str:
    """
    Convert a file to markdown using markitdown library.

    Args:
        file_path: Path to the file to convert.

    Returns:
        Markdown text.
    """
    try:
        md = MarkItDown()
        result = md.convert(str(file_path))
        return result.text_content
    except Exception as e:
        return f"Error converting file to markdown: {str(e)}"
