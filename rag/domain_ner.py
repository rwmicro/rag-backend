"""
Domain-Specific Named Entity Recognition Enhancement

Extends standard NER with domain-specific entity extraction for:
- Medical/Healthcare: diseases, medications, procedures, symptoms
- Legal: case names, statutes, legal terms
- Financial: financial instruments, metrics, companies
- Technical: technologies, programming languages, protocols
- Academic: research topics, methodologies, institutions
"""

from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from loguru import logger

from .graph_rag import Entity
from .chunking import Chunk


class Domain(Enum):
    """Supported domains"""
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    SCIENTIFIC = "scientific"


@dataclass
class DomainPattern:
    """Pattern for domain-specific entity extraction"""
    pattern: str  # Regex pattern
    entity_type: str  # Entity type label
    domain: Domain
    priority: int = 1  # Higher priority patterns checked first


class DomainSpecificNER:
    """
    Domain-specific NER enhancement

    Complements general NER (spaCy) with domain-specific patterns and dictionaries.
    Can operate standalone or enhance spaCy results.
    """

    def __init__(
        self,
        domain: Domain = Domain.GENERAL,
        custom_patterns: Optional[List[DomainPattern]] = None,
        custom_dictionaries: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Initialize domain-specific NER

        Args:
            domain: Primary domain for extraction
            custom_patterns: Additional custom patterns
            custom_dictionaries: Custom entity dictionaries
        """
        self.domain = domain
        self.patterns: List[DomainPattern] = []
        self.dictionaries: Dict[str, Set[str]] = {}

        # Load built-in patterns and dictionaries for domain
        self._load_domain_patterns()
        self._load_domain_dictionaries()

        # Add custom patterns
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        # Add custom dictionaries
        if custom_dictionaries:
            for entity_type, entities in custom_dictionaries.items():
                if entity_type in self.dictionaries:
                    self.dictionaries[entity_type].update(entities)
                else:
                    self.dictionaries[entity_type] = set(entities)

        # Sort patterns by priority
        self.patterns.sort(key=lambda p: p.priority, reverse=True)

        logger.info(
            f"Initialized DomainSpecificNER for {domain.value} "
            f"({len(self.patterns)} patterns, "
            f"{sum(len(d) for d in self.dictionaries.values())} dictionary entries)"
        )

    def _load_domain_patterns(self):
        """Load built-in patterns for domain"""

        # Medical Domain
        if self.domain == Domain.MEDICAL:
            self.patterns.extend([
                # Medications (common suffixes)
                DomainPattern(
                    pattern=r'\b\w+(?:mycin|cillin|cycline|oxacin|azole|olol|pril|sartan|statin)\b',
                    entity_type='MEDICATION',
                    domain=Domain.MEDICAL,
                    priority=3
                ),
                # Diseases/Conditions
                DomainPattern(
                    pattern=r'\b(?:cancer|diabetes|hypertension|infection|syndrome|disease|disorder)\b',
                    entity_type='DISEASE',
                    domain=Domain.MEDICAL,
                    priority=2
                ),
                # Medical procedures
                DomainPattern(
                    pattern=r'\b(?:surgery|biopsy|MRI|CT scan|X-ray|ultrasound|endoscopy|angioplasty)\b',
                    entity_type='PROCEDURE',
                    domain=Domain.MEDICAL,
                    priority=2
                ),
                # Symptoms
                DomainPattern(
                    pattern=r'\b(?:fever|pain|nausea|fatigue|headache|dizziness|bleeding)\b',
                    entity_type='SYMPTOM',
                    domain=Domain.MEDICAL,
                    priority=1
                ),
                # Lab values
                DomainPattern(
                    pattern=r'\b(?:WBC|RBC|hemoglobin|glucose|cholesterol|creatinine|ALT|AST)\s*[:<>=]\s*[\d.]+',
                    entity_type='LAB_VALUE',
                    domain=Domain.MEDICAL,
                    priority=3
                ),
            ])

        # Legal Domain
        elif self.domain == Domain.LEGAL:
            self.patterns.extend([
                # Case citations
                DomainPattern(
                    pattern=r'\b(?:[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+|\d+\s+U\.S\.\s+\d+)',
                    entity_type='CASE_NAME',
                    domain=Domain.LEGAL,
                    priority=3
                ),
                # Statutes
                DomainPattern(
                    pattern=r'\b\d+\s+U\.S\.C\.\s*§?\s*\d+(?:\([a-z]\d*\))?',
                    entity_type='STATUTE',
                    domain=Domain.LEGAL,
                    priority=3
                ),
                # Legal documents
                DomainPattern(
                    pattern=r'\b(?:contract|agreement|deed|will|trust|motion|complaint|brief)\b',
                    entity_type='LEGAL_DOCUMENT',
                    domain=Domain.LEGAL,
                    priority=2
                ),
                # Legal terms
                DomainPattern(
                    pattern=r'\b(?:plaintiff|defendant|appellant|appellee|jurisdiction|precedent)\b',
                    entity_type='LEGAL_TERM',
                    domain=Domain.LEGAL,
                    priority=1
                ),
            ])

        # Financial Domain
        elif self.domain == Domain.FINANCIAL:
            self.patterns.extend([
                # Ticker symbols
                DomainPattern(
                    pattern=r'\b[A-Z]{2,5}\b(?=\s*(?:stock|shares|equity))',
                    entity_type='TICKER',
                    domain=Domain.FINANCIAL,
                    priority=3
                ),
                # Financial metrics
                DomainPattern(
                    pattern=r'\b(?:EBITDA|P/E|EPS|ROI|ROE|revenue|profit|loss|assets|liabilities)\b',
                    entity_type='FINANCIAL_METRIC',
                    domain=Domain.FINANCIAL,
                    priority=2
                ),
                # Currency amounts
                DomainPattern(
                    pattern=r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|T))?',
                    entity_type='CURRENCY',
                    domain=Domain.FINANCIAL,
                    priority=3
                ),
                # Financial instruments
                DomainPattern(
                    pattern=r'\b(?:bond|stock|equity|derivative|option|future|swap|security)\b',
                    entity_type='FINANCIAL_INSTRUMENT',
                    domain=Domain.FINANCIAL,
                    priority=2
                ),
            ])

        # Technical Domain
        elif self.domain == Domain.TECHNICAL:
            self.patterns.extend([
                # Programming languages
                DomainPattern(
                    pattern=r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|Rust|Go|Ruby|PHP|Swift|Kotlin)\b',
                    entity_type='PROGRAMMING_LANGUAGE',
                    domain=Domain.TECHNICAL,
                    priority=2
                ),
                # Frameworks/Libraries
                DomainPattern(
                    pattern=r'\b(?:React|Vue|Angular|Django|Flask|Spring|TensorFlow|PyTorch|Keras)\b',
                    entity_type='FRAMEWORK',
                    domain=Domain.TECHNICAL,
                    priority=2
                ),
                # Protocols
                DomainPattern(
                    pattern=r'\b(?:HTTP|HTTPS|FTP|SSH|TCP|UDP|IP|DNS|SMTP|WebSocket)\b',
                    entity_type='PROTOCOL',
                    domain=Domain.TECHNICAL,
                    priority=2
                ),
                # APIs/Endpoints
                DomainPattern(
                    pattern=r'/api/v?\d+/[\w/]+',
                    entity_type='API_ENDPOINT',
                    domain=Domain.TECHNICAL,
                    priority=3
                ),
                # Version numbers
                DomainPattern(
                    pattern=r'\bv?\d+\.\d+(?:\.\d+)?(?:-(?:alpha|beta|rc)\.\d+)?',
                    entity_type='VERSION',
                    domain=Domain.TECHNICAL,
                    priority=1
                ),
            ])

        # Academic/Scientific Domain
        elif self.domain in [Domain.ACADEMIC, Domain.SCIENTIFIC]:
            self.patterns.extend([
                # Research methodologies
                DomainPattern(
                    pattern=r'\b(?:RCT|meta-analysis|systematic review|case study|survey|experiment)\b',
                    entity_type='METHODOLOGY',
                    domain=Domain.ACADEMIC,
                    priority=2
                ),
                # Statistical terms
                DomainPattern(
                    pattern=r'\b(?:p-value|confidence interval|standard deviation|correlation|regression)\b',
                    entity_type='STATISTICAL_TERM',
                    domain=Domain.ACADEMIC,
                    priority=2
                ),
                # Academic institutions
                DomainPattern(
                    pattern=r'\b(?:University of|Institute of|College of)\s+[A-Z][a-z]+',
                    entity_type='INSTITUTION',
                    domain=Domain.ACADEMIC,
                    priority=3
                ),
                # Citations (simple pattern)
                DomainPattern(
                    pattern=r'\([A-Z][a-z]+(?:\s+et al\.)?,\s*\d{4}\)',
                    entity_type='CITATION',
                    domain=Domain.ACADEMIC,
                    priority=2
                ),
            ])

    def _load_domain_dictionaries(self):
        """Load built-in entity dictionaries for domain"""

        # Medical dictionary (sample - would be much larger in production)
        if self.domain == Domain.MEDICAL:
            self.dictionaries['DISEASE'] = {
                'COVID-19', 'influenza', 'pneumonia', 'tuberculosis',
                'malaria', 'HIV', 'AIDS', 'Alzheimer\'s', 'Parkinson\'s',
                'asthma', 'COPD', 'arthritis', 'osteoporosis'
            }
            self.dictionaries['MEDICATION'] = {
                'aspirin', 'ibuprofen', 'acetaminophen', 'amoxicillin',
                'metformin', 'insulin', 'warfarin', 'prednisone'
            }

        # Financial dictionary
        elif self.domain == Domain.FINANCIAL:
            self.dictionaries['FINANCIAL_TERM'] = {
                'IPO', 'merger', 'acquisition', 'dividend', 'stock split',
                'earnings call', 'SEC filing', '10-K', '10-Q', '8-K',
                'balance sheet', 'income statement', 'cash flow'
            }

        # Technical dictionary
        elif self.domain == Domain.TECHNICAL:
            self.dictionaries['TECHNOLOGY'] = {
                'API', 'REST', 'GraphQL', 'microservices', 'serverless',
                'container', 'Kubernetes', 'Docker', 'CI/CD',
                'machine learning', 'deep learning', 'neural network'
            }

    def extract_entities(
        self,
        chunk: Chunk,
        base_entities: Optional[List[Entity]] = None,
    ) -> List[Entity]:
        """
        Extract domain-specific entities from chunk

        Args:
            chunk: Document chunk
            base_entities: Existing entities from general NER (will be augmented)

        Returns:
            List of entities (base + domain-specific)
        """
        entities = list(base_entities) if base_entities else []
        text = chunk.content

        # Extract using patterns
        for pattern_def in self.patterns:
            matches = re.finditer(pattern_def.pattern, text, re.IGNORECASE)

            for match in matches:
                entity_text = match.group(0)

                # Skip very short matches
                if len(entity_text) < 2:
                    continue

                # Check if entity already exists
                existing = next(
                    (e for e in entities if e.name.lower() == entity_text.lower()),
                    None
                )

                if existing:
                    # Update existing entity
                    if entity_text not in existing.mentions:
                        existing.mentions.append(entity_text)
                    existing.chunk_ids.add(chunk.chunk_id)
                else:
                    # Create new entity
                    entities.append(Entity(
                        name=entity_text,
                        entity_type=pattern_def.entity_type,
                        mentions=[entity_text],
                        chunk_ids={chunk.chunk_id}
                    ))

        # Extract using dictionaries
        text_lower = text.lower()
        for entity_type, entity_set in self.dictionaries.items():
            for entity_name in entity_set:
                # Case-insensitive search
                if entity_name.lower() in text_lower:
                    # Check if already exists
                    existing = next(
                        (e for e in entities if e.name.lower() == entity_name.lower()),
                        None
                    )

                    if existing:
                        existing.chunk_ids.add(chunk.chunk_id)
                        if entity_name not in existing.mentions:
                            existing.mentions.append(entity_name)
                    else:
                        entities.append(Entity(
                            name=entity_name,
                            entity_type=entity_type,
                            mentions=[entity_name],
                            chunk_ids={chunk.chunk_id}
                        ))

        logger.debug(
            f"Domain NER extracted {len(entities) - (len(base_entities) if base_entities else 0)} "
            f"additional entities from chunk {chunk.chunk_id}"
        )

        return entities

    def add_custom_pattern(
        self,
        pattern: str,
        entity_type: str,
        priority: int = 1,
    ):
        """
        Add custom extraction pattern

        Args:
            pattern: Regex pattern
            entity_type: Entity type label
            priority: Pattern priority (higher = checked first)
        """
        self.patterns.append(DomainPattern(
            pattern=pattern,
            entity_type=entity_type,
            domain=self.domain,
            priority=priority
        ))

        # Re-sort by priority
        self.patterns.sort(key=lambda p: p.priority, reverse=True)

        logger.info(f"Added custom pattern: {entity_type} (priority={priority})")

    def add_custom_dictionary(
        self,
        entity_type: str,
        entities: Set[str],
    ):
        """
        Add custom entity dictionary

        Args:
            entity_type: Entity type label
            entities: Set of entity names
        """
        if entity_type in self.dictionaries:
            self.dictionaries[entity_type].update(entities)
        else:
            self.dictionaries[entity_type] = set(entities)

        logger.info(
            f"Added custom dictionary: {entity_type} "
            f"({len(entities)} entities)"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get NER statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "domain": self.domain.value,
            "pattern_count": len(self.patterns),
            "entity_types": list(set(p.entity_type for p in self.patterns)),
            "dictionary_types": list(self.dictionaries.keys()),
            "total_dictionary_entries": sum(len(d) for d in self.dictionaries.values()),
        }


class HybridNER:
    """
    Hybrid NER combining general NER (spaCy) with domain-specific NER

    Provides best of both approaches:
    - General NER: Broad coverage, pretrained models
    - Domain NER: Specialized patterns, custom dictionaries
    """

    def __init__(
        self,
        general_ner=None,  # spaCy NLP pipeline or GraphRAG instance
        domain_ner: Optional[DomainSpecificNER] = None,
        merge_strategy: str = "union",  # "union", "domain_first", "general_first"
    ):
        """
        Initialize hybrid NER

        Args:
            general_ner: General NER system (spaCy or GraphRAG)
            domain_ner: Domain-specific NER
            merge_strategy: How to merge results
        """
        self.general_ner = general_ner
        self.domain_ner = domain_ner
        self.merge_strategy = merge_strategy

        logger.info(f"Initialized HybridNER (strategy={merge_strategy})")

    def extract_entities(self, chunk: Chunk) -> List[Entity]:
        """
        Extract entities using hybrid approach

        Args:
            chunk: Document chunk

        Returns:
            Merged list of entities
        """
        all_entities = []

        # Extract with general NER
        if self.general_ner:
            if hasattr(self.general_ner, '_extract_entities'):
                # GraphRAG instance
                general_entities = self.general_ner._extract_entities(chunk)
            else:
                # Assume spaCy NLP pipeline
                general_entities = self._extract_with_spacy(chunk, self.general_ner)

            all_entities.extend(general_entities)

        # Extract with domain NER
        if self.domain_ner:
            if self.merge_strategy == "union":
                # Augment general entities with domain-specific
                domain_entities = self.domain_ner.extract_entities(
                    chunk,
                    base_entities=all_entities
                )
                all_entities = domain_entities
            elif self.merge_strategy == "domain_first":
                # Domain entities first, then general
                domain_entities = self.domain_ner.extract_entities(chunk)
                all_entities = domain_entities + all_entities
            else:  # general_first
                # General entities first, then domain
                domain_entities = self.domain_ner.extract_entities(chunk)
                all_entities = all_entities + domain_entities

        # Remove duplicates (keep first occurrence)
        seen = set()
        unique_entities = []
        for entity in all_entities:
            entity_key = entity.name.lower()
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)

        return unique_entities

    def _extract_with_spacy(self, chunk: Chunk, nlp) -> List[Entity]:
        """Extract entities using spaCy NLP pipeline"""
        entities = []
        doc = nlp(chunk.content)

        for ent in doc.ents:
            entities.append(Entity(
                name=ent.text,
                entity_type=ent.label_,
                mentions=[ent.text],
                chunk_ids={chunk.chunk_id}
            ))

        return entities
