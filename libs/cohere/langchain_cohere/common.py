from dataclasses import dataclass
from typing import Any, List, Mapping, Set


@dataclass
class CohereCitation:
    """
    Cohere has fine-grained citations that specify the exact part of text.
    More info at https://docs.cohere.com/docs/documents-and-citations
    """

    """
    The index of text that the citation starts at, counting from zero. For example, a 
    generation of 'Hello, world!' with a citation on 'world' would have a start value 
    of 7.  This is because the citation starts at 'w', which is the seventh character.
    """
    start: int

    """
    The index of text that the citation ends after, counting from zero. For example, a 
    generation of 'Hello, world!' with a citation on 'world' would have an end value of
    11. This is because the citation ends after 'd', which is the eleventh character.
    """
    end: int

    """
    The text of the citation. For example, a generation of 'Hello, world!' with a 
    citation of 'world' would have a text value of 'world'.
    """
    text: str

    """
    The contents of the documents that were cited. When used with agents these will be 
    the contents of relevant agent outputs.
    
    Every document will have a string field called `id`. This can be used with the 
    `document_ids` field to deduplicate documents across several citations. The `id` 
    field is created on the document when it doesn't already exist.
    """
    documents: List[Mapping[str, Any]]

    """
    A set of the `id` field from all the documents in the `documents` field. 
    """
    document_ids: Set[str]
