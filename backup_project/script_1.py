# Now let's create the course processing pipeline
class CourseProcessor:
    """
    Robust course processing pipeline for cleaning, normalizing, and extracting 
    metadata from course descriptions and catalog data.
    """
    
    def __init__(self):
        print("Initializing Course Processing Pipeline...")
        print("Features:")
        print("- Text preprocessing and normalization")
        print("- Metadata extraction (credits, prerequisites)")
        print("- Learning outcomes parsing")
        print("- Batch processing support")
        print("- Input validation and sanitization")
        print("- Progress tracking for large datasets")
        
        # Text processing configuration
        self.processing_config = {
            'min_description_length': 20,
            'max_description_length': 2000,
            'required_fields': ['course_id', 'title', 'description'],
            'optional_fields': ['credits', 'prerequisites', 'learning_outcomes', 'level', 'domain']
        }
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'errors': 0,
            'warnings': 0
        }
        
    def validate_course_data(self, course_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate course data structure and content
        
        Args:
            course_data: Raw course data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.processing_config['required_fields']:
            if field not in course_data or not course_data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate description length
        desc = course_data.get('description', '')
        if len(desc) < self.processing_config['min_description_length']:
            errors.append(f"Description too short (minimum {self.processing_config['min_description_length']} characters)")
        elif len(desc) > self.processing_config['max_description_length']:
            errors.append(f"Description too long (maximum {self.processing_config['max_description_length']} characters)")
        
        # Validate course ID format
        course_id = course_data.get('course_id', '')
        if not re.match(r'^[A-Z]{2,6}\d{1,4}[A-Z]?$', course_id):
            errors.append("Course ID format invalid (expected format: CS101, MATH200A, etc.)")
        
        # Validate credits
        credits = course_data.get('credits', 0)
        if credits and (not isinstance(credits, (int, float)) or credits < 0 or credits > 20):
            errors.append("Credits must be a number between 0 and 20")
        
        return len(errors) == 0, errors
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize course description text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', cleaned)
        
        # Normalize common academic terms
        replacements = {
            r'\b(pre-req|prereq|pre req)\b': 'prerequisite',
            r'\b(co-req|coreq|co req)\b': 'corequisite',
            r'\bcrs?\b': 'course',
            r'\bdept\b': 'department',
            r'\buniv\b': 'university'
        }
        
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def extract_credits(self, text: str, existing_credits: Optional[Union[int, float]] = None) -> float:
        """
        Extract credit information from course description or metadata
        
        Args:
            text: Course description text
            existing_credits: Already provided credit value
            
        Returns:
            Number of credits (default 3.0 if not found)
        """
        if existing_credits is not None and existing_credits > 0:
            return float(existing_credits)
        
        # Common credit patterns in course descriptions
        credit_patterns = [
            r'(\d+(?:\.\d+)?)\s*credits?',
            r'(\d+(?:\.\d+)?)\s*credit\s*hours?',
            r'(\d+(?:\.\d+)?)\s*hrs?',
            r'(\d+(?:\.\d+)?)\s*units?'
        ]
        
        for pattern in credit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Default to 3 credits if not found
        return 3.0
    
    def extract_prerequisites(self, text: str, existing_prereqs: Optional[List] = None) -> List[str]:
        """
        Extract prerequisite information from course description
        
        Args:
            text: Course description text
            existing_prereqs: Already provided prerequisite list
            
        Returns:
            List of prerequisite course codes
        """
        if existing_prereqs:
            return existing_prereqs
        
        prerequisites = []
        
        # Find prerequisite sections
        prereq_patterns = [
            r'prerequisite[s]?:?\s*([^.]+)',
            r'pre-?req[s]?:?\s*([^.]+)',
            r'requires?:?\s*([^.]+)',
            r'must\s+have\s+completed:?\s*([^.]+)'
        ]
        
        for pattern in prereq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract course codes from the prerequisite text
                course_codes = re.findall(r'[A-Z]{2,6}\s*\d{1,4}[A-Z]?', match, re.IGNORECASE)
                prerequisites.extend([code.replace(' ', '') for code in course_codes])
        
        return list(set(prerequisites))  # Remove duplicates
    
    def extract_learning_outcomes(self, text: str, existing_outcomes: Optional[List] = None) -> List[str]:
        """
        Extract learning outcomes from course description
        
        Args:
            text: Course description text
            existing_outcomes: Already provided outcomes list
            
        Returns:
            List of learning outcomes
        """
        if existing_outcomes:
            return existing_outcomes
        
        outcomes = []
        
        # Patterns for learning outcomes sections
        outcome_patterns = [
            r'(?:learning\s+)?outcomes?:?\s*([^.]*(?:\.[^.]*)*)',
            r'objectives?:?\s*([^.]*(?:\.[^.]*)*)',
            r'students?\s+will\s+(?:be\s+able\s+to\s*)?:?\s*([^.]*(?:\.[^.]*)*)',
            r'upon\s+completion[^.]*students?\s+will\s*:?\s*([^.]*(?:\.[^.]*)*)'
        ]
        
        for pattern in outcome_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by common separators and clean
                outcome_items = re.split(r'[;\n\r]|\s*\*\s*|\s*-\s*|\s*\d+\.\s*', match)
                for item in outcome_items:
                    item = item.strip()
                    if len(item) > 10:  # Filter out very short items
                        outcomes.append(item)
        
        return outcomes[:10]  # Limit to reasonable number
    
    def extract_course_level(self, course_id: str, description: str) -> str:
        """
        Determine course level (undergraduate/graduate) from course code and description
        
        Args:
            course_id: Course identifier (e.g., CS101, MATH500)
            description: Course description text
            
        Returns:
            Course level string
        """
        # Extract number from course ID
        number_match = re.search(r'\d+', course_id)
        if number_match:
            course_number = int(number_match.group())
            
            if course_number < 300:
                return 'undergraduate_lower'
            elif course_number < 500:
                return 'undergraduate_upper'
            elif course_number < 700:
                return 'graduate'
            else:
                return 'doctoral'
        
        # Check description for level indicators
        grad_keywords = ['graduate', 'advanced', 'research', 'thesis', 'dissertation']
        undergrad_keywords = ['introductory', 'fundamental', 'basic', 'elementary']
        
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in grad_keywords):
            return 'graduate'
        elif any(keyword in description_lower for keyword in undergrad_keywords):
            return 'undergraduate_lower'
        
        return 'undergraduate'
    
    def extract_domain(self, course_id: str, title: str, description: str) -> str:
        """
        Determine course domain/subject area from course information
        
        Args:
            course_id: Course identifier
            title: Course title
            description: Course description
            
        Returns:
            Domain classification string
        """
        # Extract department code from course ID
        dept_match = re.match(r'^([A-Z]+)', course_id)
        if dept_match:
            dept_code = dept_match.group(1)
            
            # Common department code mappings
            dept_mappings = {
                'CS': 'computer_science',
                'COMP': 'computer_science',
                'MATH': 'mathematics',
                'STAT': 'statistics',
                'PHYS': 'physics',
                'CHEM': 'chemistry',
                'BIOL': 'biology',
                'ENG': 'english',
                'HIST': 'history',
                'PSYC': 'psychology',
                'ECON': 'economics',
                'POLS': 'political_science',
                'PHIL': 'philosophy',
                'ART': 'art',
                'MUS': 'music',
                'PE': 'physical_education'
            }
            
            if dept_code in dept_mappings:
                return dept_mappings[dept_code]
        
        # Fallback to keyword analysis
        combined_text = f"{title} {description}".lower()
        
        domain_keywords = {
            'computer_science': ['programming', 'algorithm', 'software', 'computing', 'data structure'],
            'mathematics': ['calculus', 'algebra', 'geometry', 'statistics', 'mathematical'],
            'physics': ['physics', 'mechanics', 'thermodynamics', 'quantum', 'electromagnetic'],
            'chemistry': ['chemistry', 'chemical', 'molecular', 'organic', 'inorganic'],
            'biology': ['biology', 'biological', 'anatomy', 'genetics', 'ecology'],
            'business': ['business', 'management', 'marketing', 'finance', 'accounting'],
            'psychology': ['psychology', 'behavioral', 'cognitive', 'social psychology']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return domain
        
        return 'general'
    
    def process_single_course(self, course_data: Dict) -> Dict:
        """
        Process a single course through the complete pipeline
        
        Args:
            course_data: Raw course data dictionary
            
        Returns:
            Processed course data with extracted metadata
        """
        self.stats['total_processed'] += 1
        
        try:
            # Validate input data
            is_valid, errors = self.validate_course_data(course_data)
            if not is_valid:
                self.stats['errors'] += 1
                return {
                    'success': False,
                    'errors': errors,
                    'original_data': course_data
                }
            
            # Extract and clean basic fields
            processed = {
                'course_id': course_data['course_id'].upper().replace(' ', ''),
                'title': self.clean_text(course_data['title']),
                'description': self.clean_text(course_data['description']),
                'description_original': course_data['description']
            }
            
            # Extract metadata
            processed['credits'] = self.extract_credits(
                processed['description'], 
                course_data.get('credits')
            )
            
            processed['prerequisites'] = self.extract_prerequisites(
                processed['description'],
                course_data.get('prerequisites')
            )
            
            processed['learning_outcomes'] = self.extract_learning_outcomes(
                processed['description'],
                course_data.get('learning_outcomes')
            )
            
            processed['level'] = self.extract_course_level(
                processed['course_id'],
                processed['description']
            )
            
            processed['domain'] = self.extract_domain(
                processed['course_id'],
                processed['title'],
                processed['description']
            )
            
            # Add processing metadata
            processed['processing_info'] = {
                'processed_at': datetime.now().isoformat(),
                'processor_version': '1.0',
                'validation_passed': True,
                'extracted_fields': ['credits', 'prerequisites', 'learning_outcomes', 'level', 'domain']
            }
            
            # Success metrics
            processed['success'] = True
            self.stats['successful'] += 1
            
            return processed
            
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'success': False,
                'errors': [str(e)],
                'original_data': course_data,
                'processing_info': {
                    'processed_at': datetime.now().isoformat(),
                    'error_type': type(e).__name__
                }
            }
    
    def process_batch(self, courses_list: List[Dict], show_progress: bool = True) -> List[Dict]:
        """
        Process multiple courses in batch with progress tracking
        
        Args:
            courses_list: List of course data dictionaries
            show_progress: Whether to show progress updates
            
        Returns:
            List of processed course dictionaries
        """
        processed_courses = []
        total = len(courses_list)
        
        print(f"\nProcessing {total} courses in batch...")
        
        for i, course_data in enumerate(courses_list):
            if show_progress and (i % 10 == 0 or i == total - 1):
                progress = (i + 1) / total * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{total})")
            
            processed = self.process_single_course(course_data)
            processed_courses.append(processed)
        
        # Print processing summary
        print(f"\nBatch processing complete!")
        print(f"✓ Total processed: {self.stats['total_processed']}")
        print(f"✓ Successful: {self.stats['successful']}")
        print(f"✗ Errors: {self.stats['errors']}")
        print(f"⚠ Success rate: {(self.stats['successful']/self.stats['total_processed']*100):.1f}%")
        
        return processed_courses
    
    def export_processed_data(self, processed_courses: List[Dict], filename: str = None) -> str:
        """
        Export processed course data to CSV format
        
        Args:
            processed_courses: List of processed course dictionaries
            filename: Output filename (optional)
            
        Returns:
            Filename of exported data
        """
        if not filename:
            filename = f"processed_courses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Prepare data for export
        export_data = []
        for course in processed_courses:
            if course.get('success', False):
                row = {
                    'course_id': course['course_id'],
                    'title': course['title'],
                    'description': course['description'],
                    'credits': course['credits'],
                    'prerequisites': ';'.join(course['prerequisites']),
                    'learning_outcomes': ';'.join(course['learning_outcomes']),
                    'level': course['level'],
                    'domain': course['domain'],
                    'processed_at': course['processing_info']['processed_at']
                }
                export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        
        print(f"Exported {len(export_data)} processed courses to {filename}")
        return filename

# Test the course processor with sample data
def test_course_processor():
    """Test the course processing pipeline"""
    
    print("\n" + "="*80)
    print("COURSE PROCESSING PIPELINE TEST")
    print("="*80)
    
    processor = CourseProcessor()
    
    # Sample course data with various formats and quality levels
    sample_courses = [
        {
            'course_id': 'CS101',
            'title': 'Introduction to Computer Science',
            'description': 'This course provides a comprehensive introduction to computer science concepts. Students will learn programming fundamentals using Python, including variables, control structures, functions, and basic data structures. Prerequisites: None. Learning outcomes: Students will be able to write basic programs, understand algorithm complexity, and solve computational problems.',
            'credits': 3
        },
        {
            'course_id': 'MATH 200',  # Note: has space
            'title': 'Calculus I',
            'description': 'Differential and integral calculus. Limits, derivatives, applications of derivatives, introduction to integration. Prerequisite: MATH099 or equivalent. 4 credit hours. Upon completion students will: calculate limits, compute derivatives, apply derivatives to optimization problems.',
        },
        {
            'course_id': 'HIST150',
            'title': 'World History',
            'description': 'Survey of world civilizations from ancient times to present. Covers major political, social, and cultural developments.',
            'credits': 3,
            'prerequisites': [],
            'learning_outcomes': [
                'Analyze historical patterns',
                'Compare different civilizations',
                'Evaluate primary sources'
            ]
        },
        {
            'course_id': 'invalid',  # Invalid course ID format
            'title': 'Bad Course',
            'description': 'Short desc',  # Too short description
            'credits': 'invalid'  # Invalid credits
        },
        {
            'course_id': 'PHYS301',
            'title': 'Quantum Mechanics',
            'description': 'Advanced course in quantum mechanics covering wave functions, operators, and quantum systems. This is a graduate-level course requiring strong mathematical background. Prerequisites: PHYS201, PHYS202, MATH301. Students will develop understanding of quantum principles and solve complex quantum mechanical problems.',
            'credits': 4
        }
    ]
    
    print(f"\nTesting with {len(sample_courses)} sample courses...")
    
    # Process individual course first
    print("\n" + "-"*50)
    print("INDIVIDUAL COURSE PROCESSING TEST")
    print("-"*50)
    
    test_course = sample_courses[0]
    processed_single = processor.process_single_course(test_course)
    
    if processed_single['success']:
        print(f"✓ Successfully processed: {processed_single['course_id']}")
        print(f"  Title: {processed_single['title']}")
        print(f"  Credits: {processed_single['credits']}")
        print(f"  Level: {processed_single['level']}")
        print(f"  Domain: {processed_single['domain']}")
        print(f"  Prerequisites: {processed_single['prerequisites']}")
        print(f"  Learning Outcomes: {len(processed_single['learning_outcomes'])} found")
    else:
        print(f"✗ Processing failed: {processed_single['errors']}")
    
    # Process batch
    print("\n" + "-"*50)
    print("BATCH PROCESSING TEST")
    print("-"*50)
    
    processed_batch = processor.process_batch(sample_courses)
    
    # Display results summary
    print("\n" + "-"*30)
    print("DETAILED RESULTS")
    print("-"*30)
    
    for i, result in enumerate(processed_batch):
        course_id = sample_courses[i].get('course_id', f'Course {i+1}')
        if result['success']:
            print(f"\n✅ {course_id}: SUCCESS")
            print(f"   Processed ID: {result['course_id']}")
            print(f"   Credits: {result['credits']}")
            print(f"   Level: {result['level']}")
            print(f"   Domain: {result['domain']}")
            print(f"   Prerequisites: {len(result['prerequisites'])}")
            print(f"   Learning Outcomes: {len(result['learning_outcomes'])}")
        else:
            print(f"\n❌ {course_id}: FAILED")
            print(f"   Errors: {'; '.join(result['errors'])}")
    
    # Export test
    print("\n" + "-"*50)
    print("DATA EXPORT TEST")
    print("-"*50)
    
    export_filename = processor.export_processed_data(processed_batch, "test_courses_processed.csv")
    
    return {
        'processor': processor,
        'processed_data': processed_batch,
        'export_file': export_filename
    }

# Run the test
processor_test_results = test_course_processor()