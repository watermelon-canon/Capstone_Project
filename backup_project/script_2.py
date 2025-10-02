# Now let's create sample course datasets for comprehensive testing
import json
import random

def generate_sample_course_datasets():
    """
    Generate comprehensive sample course datasets for testing the system
    Includes courses from multiple domains with realistic data
    """
    
    print("="*80)
    print("GENERATING SAMPLE COURSE DATASETS")
    print("="*80)
    
    # Computer Science Courses
    cs_courses = [
        {
            'course_id': 'CS101',
            'title': 'Introduction to Programming',
            'description': 'Fundamental programming concepts using Python. Topics include variables, control structures, functions, lists, dictionaries, and basic algorithms. Students will develop problem-solving skills and learn to write, debug, and test programs. Prerequisites: None. 3 credit hours.',
            'credits': 3,
            'prerequisites': [],
            'learning_outcomes': [
                'Write programs using Python programming language',
                'Apply programming concepts to solve computational problems',
                'Debug and test software programs',
                'Understand basic data structures and algorithms'
            ],
            'level': 'undergraduate',
            'domain': 'computer_science'
        },
        {
            'course_id': 'CS102',
            'title': 'Data Structures and Algorithms',
            'description': 'Study of fundamental data structures including arrays, linked lists, stacks, queues, trees, and hash tables. Analysis of algorithm complexity and design techniques. Implementation in Java. Prerequisites: CS101. 4 credit hours.',
            'credits': 4,
            'prerequisites': ['CS101'],
            'learning_outcomes': [
                'Implement fundamental data structures',
                'Analyze algorithm time and space complexity',
                'Design efficient algorithms for common problems',
                'Choose appropriate data structures for specific applications'
            ],
            'level': 'undergraduate',
            'domain': 'computer_science'
        },
        {
            'course_id': 'CS301',
            'title': 'Database Systems',
            'description': 'Design and implementation of database systems. Relational model, SQL, normalization, transactions, and concurrent access. Introduction to NoSQL databases and distributed systems. Prerequisites: CS102. 3 credit hours.',
            'credits': 3,
            'prerequisites': ['CS102'],
            'learning_outcomes': [
                'Design relational database schemas',
                'Write complex SQL queries',
                'Understand database normalization principles',
                'Implement database applications'
            ],
            'level': 'undergraduate',
            'domain': 'computer_science'
        },
        {
            'course_id': 'CS401',
            'title': 'Machine Learning',
            'description': 'Introduction to machine learning algorithms and applications. Supervised and unsupervised learning, neural networks, deep learning, and model evaluation. Hands-on projects using Python and scikit-learn. Prerequisites: CS102, MATH201, STAT200. 4 credit hours.',
            'credits': 4,
            'prerequisites': ['CS102', 'MATH201', 'STAT200'],
            'learning_outcomes': [
                'Apply machine learning algorithms to real-world problems',
                'Evaluate and compare different learning models',
                'Implement neural networks and deep learning systems',
                'Understand statistical foundations of machine learning'
            ],
            'level': 'undergraduate',
            'domain': 'computer_science'
        }
    ]
    
    # Mathematics Courses
    math_courses = [
        {
            'course_id': 'MATH101',
            'title': 'College Algebra',
            'description': 'Review of algebraic concepts including polynomials, rational functions, exponential and logarithmic functions. Systems of equations and inequalities. Prerequisite: High school algebra or placement test. 3 credit hours.',
            'credits': 3,
            'prerequisites': [],
            'learning_outcomes': [
                'Solve polynomial and rational equations',
                'Work with exponential and logarithmic functions',
                'Solve systems of linear equations',
                'Apply algebraic concepts to word problems'
            ],
            'level': 'undergraduate',
            'domain': 'mathematics'
        },
        {
            'course_id': 'MATH201',
            'title': 'Calculus I',
            'description': 'Differential calculus of functions of one variable. Limits, continuity, derivatives, and applications including optimization and related rates. Introduction to integration. Prerequisites: MATH101 or equivalent. 4 credit hours.',
            'credits': 4,
            'prerequisites': ['MATH101'],
            'learning_outcomes': [
                'Calculate limits and determine continuity',
                'Compute derivatives using various techniques',
                'Apply derivatives to solve optimization problems',
                'Understand the relationship between derivatives and integrals'
            ],
            'level': 'undergraduate',
            'domain': 'mathematics'
        },
        {
            'course_id': 'MATH202',
            'title': 'Calculus II',
            'description': 'Integral calculus and infinite series. Techniques of integration, applications of integrals, parametric equations, polar coordinates, and convergence tests for series. Prerequisites: MATH201. 4 credit hours.',
            'credits': 4,
            'prerequisites': ['MATH201'],
            'learning_outcomes': [
                'Master various techniques of integration',
                'Apply integrals to solve geometric and physical problems',
                'Work with parametric and polar coordinate systems',
                'Analyze convergence of infinite series'
            ],
            'level': 'undergraduate',
            'domain': 'mathematics'
        }
    ]
    
    # Business Courses
    business_courses = [
        {
            'course_id': 'BUS101',
            'title': 'Introduction to Business',
            'description': 'Overview of business principles including management, marketing, finance, and operations. Introduction to entrepreneurship and business ethics. Case studies of successful businesses. No prerequisites. 3 credit hours.',
            'credits': 3,
            'prerequisites': [],
            'learning_outcomes': [
                'Understand fundamental business concepts',
                'Analyze business case studies',
                'Evaluate ethical considerations in business decisions',
                'Identify entrepreneurial opportunities'
            ],
            'level': 'undergraduate',
            'domain': 'business'
        },
        {
            'course_id': 'BUS201',
            'title': 'Financial Accounting',
            'description': 'Principles of financial accounting including the accounting cycle, financial statements, and analysis techniques. Emphasis on recording, summarizing, and interpreting financial data. Prerequisites: BUS101. 3 credit hours.',
            'credits': 3,
            'prerequisites': ['BUS101'],
            'learning_outcomes': [
                'Prepare basic financial statements',
                'Understand the accounting cycle',
                'Analyze financial performance using ratios',
                'Apply generally accepted accounting principles'
            ],
            'level': 'undergraduate',
            'domain': 'business'
        }
    ]
    
    # Psychology Courses
    psychology_courses = [
        {
            'course_id': 'PSYC101',
            'title': 'General Psychology',
            'description': 'Introduction to psychological science covering behavior, cognition, development, personality, and mental health. Research methods and statistical analysis in psychology. No prerequisites. 3 credit hours.',
            'credits': 3,
            'prerequisites': [],
            'learning_outcomes': [
                'Understand major psychological theories and concepts',
                'Evaluate psychological research methods',
                'Apply psychological principles to real-world situations',
                'Analyze human behavior from multiple perspectives'
            ],
            'level': 'undergraduate',
            'domain': 'psychology'
        },
        {
            'course_id': 'PSYC201',
            'title': 'Research Methods in Psychology',
            'description': 'Scientific methods in psychological research including experimental design, data collection, and statistical analysis. Laboratory experience with research projects. Prerequisites: PSYC101, STAT200. 4 credit hours.',
            'credits': 4,
            'prerequisites': ['PSYC101', 'STAT200'],
            'learning_outcomes': [
                'Design psychological experiments',
                'Collect and analyze psychological data',
                'Write research reports in APA format',
                'Evaluate the validity of psychological studies'
            ],
            'level': 'undergraduate',
            'domain': 'psychology'
        }
    ]
    
    # Combine all courses
    all_courses = cs_courses + math_courses + business_courses + psychology_courses
    
    print(f"Generated {len(all_courses)} sample courses:")
    print(f"  • Computer Science: {len(cs_courses)} courses")
    print(f"  • Mathematics: {len(math_courses)} courses") 
    print(f"  • Business: {len(business_courses)} courses")
    print(f"  • Psychology: {len(psychology_courses)} courses")
    
    # Create alternative versions of some courses (for equivalency testing)
    alternative_courses = [
        {
            'course_id': 'COMP110',
            'title': 'Programming Fundamentals',
            'description': 'Introduction to computer programming using Python. Covers basic programming concepts including variables, loops, conditionals, functions, and simple data structures. Problem-solving techniques and algorithm development. No prerequisites. 3 credit hours.',
            'credits': 3,
            'prerequisites': [],
            'learning_outcomes': [
                'Develop programs using Python programming language',
                'Understand fundamental programming concepts',
                'Apply problem-solving techniques to programming challenges',
                'Debug and test computer programs effectively'
            ],
            'level': 'undergraduate',
            'domain': 'computer_science'
        },
        {
            'course_id': 'MATH150',
            'title': 'Calculus with Applications',
            'description': 'Single-variable calculus with emphasis on applications to business and life sciences. Derivatives, optimization, integration, and elementary differential equations. Prerequisites: College algebra or equivalent. 4 credit hours.',
            'credits': 4,
            'prerequisites': ['MATH101'],
            'learning_outcomes': [
                'Compute derivatives and integrals of functions',
                'Solve optimization problems in applied contexts',
                'Apply calculus to business and scientific problems',
                'Model real-world situations using calculus concepts'
            ],
            'level': 'undergraduate',
            'domain': 'mathematics'
        },
        {
            'course_id': 'MGMT101',
            'title': 'Foundations of Business',
            'description': 'Survey of business fundamentals including organizational behavior, marketing strategies, financial management, and operations. Exploration of contemporary business challenges and opportunities. No prerequisites. 3 credit hours.',
            'credits': 3,
            'prerequisites': [],
            'learning_outcomes': [
                'Analyze contemporary business practices',
                'Understand organizational structures and management',
                'Evaluate marketing and financial strategies',
                'Assess business opportunities and challenges'
            ],
            'level': 'undergraduate',
            'domain': 'business'
        }
    ]
    
    print(f"Generated {len(alternative_courses)} alternative course versions for equivalency testing")
    
    # Create datasets with different characteristics for testing
    datasets = {
        'primary_catalog': all_courses,
        'alternative_catalog': alternative_courses,
        'combined_catalog': all_courses + alternative_courses,
        'cs_only': cs_courses,
        'math_only': math_courses
    }
    
    # Export datasets to files
    exported_files = []
    
    for dataset_name, courses in datasets.items():
        filename = f"sample_courses_{dataset_name}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(courses, f, indent=2, ensure_ascii=False)
        
        exported_files.append(filename)
        print(f"✓ Exported {len(courses)} courses to {filename}")
    
    # Create CSV versions for easier analysis
    csv_files = []
    for dataset_name, courses in datasets.items():
        # Convert to DataFrame format
        df_data = []
        for course in courses:
            row = {
                'course_id': course['course_id'],
                'title': course['title'],
                'description': course['description'],
                'credits': course['credits'],
                'prerequisites': '; '.join(course['prerequisites']),
                'learning_outcomes': '; '.join(course['learning_outcomes']),
                'level': course['level'],
                'domain': course['domain']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_filename = f"sample_courses_{dataset_name}.csv"
        df.to_csv(csv_filename, index=False)
        csv_files.append(csv_filename)
        print(f"✓ Exported {len(courses)} courses to {csv_filename}")
    
    # Generate statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    all_domains = [course['domain'] for course in all_courses]
    domain_counts = pd.Series(all_domains).value_counts()
    
    print("\nCourses by Domain:")
    for domain, count in domain_counts.items():
        print(f"  • {domain.replace('_', ' ').title()}: {count}")
    
    all_levels = [course['level'] for course in all_courses]
    level_counts = pd.Series(all_levels).value_counts()
    
    print("\nCourses by Level:")
    for level, count in level_counts.items():
        print(f"  • {level.replace('_', ' ').title()}: {count}")
    
    credit_distribution = [course['credits'] for course in all_courses]
    print(f"\nCredit Distribution:")
    print(f"  • Average credits: {np.mean(credit_distribution):.1f}")
    print(f"  • Credit range: {min(credit_distribution)}-{max(credit_distribution)}")
    
    prereq_counts = [len(course['prerequisites']) for course in all_courses]
    print(f"\nPrerequisite Analysis:")
    print(f"  • Average prerequisites: {np.mean(prereq_counts):.1f}")
    print(f"  • Courses with no prerequisites: {prereq_counts.count(0)}")
    print(f"  • Maximum prerequisites: {max(prereq_counts)}")
    
    return {
        'datasets': datasets,
        'json_files': exported_files,
        'csv_files': csv_files,
        'statistics': {
            'total_courses': len(all_courses),
            'domains': dict(domain_counts),
            'levels': dict(level_counts),
            'avg_credits': np.mean(credit_distribution),
            'avg_prerequisites': np.mean(prereq_counts)
        }
    }

# Generate the sample datasets
sample_data = generate_sample_course_datasets()