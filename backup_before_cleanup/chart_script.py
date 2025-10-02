# Create a mermaid flowchart for the Course Equivalency Mapping System architecture
diagram_code = '''
flowchart TD
    subgraph IL ["🔢 Input Layer"]
        CC["Course Catalogs<br/>CSV/JSON"]
        SD["Syllabi & Descriptions<br/>Text Files"]
        LO["Learning Outcomes<br/>Structured Data"]
    end
    
    subgraph PL ["⚙️ Processing Layer"]
        TP["Text Preprocessing<br/>spaCy/NLTK"]
        FE["Feature Extraction<br/>scikit-learn"]
        DV["Data Validation<br/>Pandas"]
    end
    
    subgraph AI ["🤖 AI/ML Layer"]
        SS["Semantic Similarity<br/>sentence-transformers"]
        GA["Graph Analysis<br/>NetworkX + PyTorch Geometric"]
        HS["Hybrid Scoring<br/>Custom Ensemble"]
    end
    
    subgraph AL ["📊 Analysis Layer"]
        SE["Similarity Engine<br/>Multi-dimensional Scoring"]
        CC2["Confidence Calculator<br/>Statistical Methods"]
        FE2["Fairness Evaluator<br/>Bias Detection"]
    end
    
    subgraph OL ["📱 Output Layer"]
        SI["Streamlit Interface<br/>Interactive Dashboard"]
        PV["Plotly Visualizations<br/>Charts & Graphs"]
        RG["Report Generation<br/>CSV/PDF Export"]
    end
    
    %% Data flow connections
    CC --> TP
    SD --> TP
    LO --> FE
    CC --> FE
    SD --> FE
    
    TP --> SS
    FE --> SS
    DV --> GA
    FE --> GA
    
    SS --> HS
    GA --> HS
    
    HS --> SE
    SS --> SE
    SE --> CC2
    SE --> FE2
    
    SE --> SI
    CC2 --> SI
    FE2 --> PV
    SE --> PV
    CC2 --> RG
    FE2 --> RG
    
    %% Additional validation flows
    CC --> DV
    SD --> DV
    LO --> DV
'''

# Create the mermaid diagram
png_path, svg_path = create_mermaid_diagram(diagram_code, 'course_equivalency_architecture.png', 'course_equivalency_architecture.svg')

print(f"System architecture diagram saved as: {png_path} and {svg_path}")