from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
from typing import Dict, Any
from datetime import datetime
import spacy
from pathlib import Path
import uuid
from PIL import Image


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Path("reports").mkdir(exist_ok=True)

try :
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

current_analysis = {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request":request})

@app.post("/analyze")
async def analyze_data(request: Request, file: UploadFile=File(...)):

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(await file.read()))
        else:
            df = pd.read_excel(BytesIO(await file.read()))

        df = df.dropna(how='all').reset_index(drop=True)

        column_types = detect_column_types(df)

        analysis = {
            "description" : df.describe(include='all').to_dict(),
            "column_types" : column_types,
            "missing_values" : df.isnull().sum().to_dict(),
            "shape" : df.shape,
            "correlation" : df.select_dtypes(include=[np.number]).corr().to_dict(),
            "data_head" : df.head().to_dict()
        }

        viz_files = generate_visualizations(df, column_types)
        analysis["visualizations"] = viz_files

        analysis_id = str(uuid.uuid4())
        current_analysis[analysis_id] = {
            "df" : df,
            "analysis" : analysis,
            "viz_files" : viz_files
        }

        return templates.TemplateResponse(
            "report.html",
            {
                "request": request,
                "analysis": analysis,
                "analysis_id" : analysis_id
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request, "error": str(e)
            },
            status_code=400
        )
    
@app.post("/report")
async def generate_report(
    request: Request,
    analysis_id: str = Form(...),
    report_format: str = Form("xlsx")
):
    """Generate and download report"""
    try:
        analysis_data = current_analysis.get(analysis_id)
        if not analysis_data:
            raise ValueError("Analysis not found or expired")
        
        df = analysis_data["df"]
        analysis = analysis_data["analysis"]
        viz_files = analysis_data["viz_files"]
        
        if report_format == "xlsx":
            report_path = f"reports/analysis_report_{analysis_id}.xlsx"
            generate_excel_report(df, analysis, viz_files, report_path)
            
            return FileResponse(
                report_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"analysis_report.xlsx"
            )
        else:
            # PDF generation would go here
            raise NotImplementedError("PDF generation not implemented yet")
            
    except Exception as e:
        return templates.TemplateResponse(
            "report.html",
            {"request": request, "analysis": analysis, "error": str(e)},
            status_code=400
        )


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column types with NLP assistance"""
    column_types = {}
    for col in df.columns:
        # Check numeric first
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
        else:
            # Use NLP for text columns
            sample_text = str(df[col].dropna().iloc[0]) if not df[col].empty else ""
            doc = nlp(sample_text[:100])  # Process first 100 chars
            
            if any(ent.label_ in ["DATE", "TIME"] for ent in doc.ents):
                column_types[col] = "datetime"
            elif len(sample_text) > 50:
                column_types[col] = "text"
            else:
                column_types[col] = "categorical"
    return column_types

def generate_visualizations(df: pd.DataFrame, column_types: Dict[str, str]):
    """Generate visualizations and return their paths"""
    viz_files = []
    temp_dir = tempfile.mkdtemp()
    
    for col, col_type in column_types.items():
        try:
            plt.figure(figsize=(10, 6))
            if col_type == "numeric":
                sns.histplot(df[col].dropna())
                plt.title(f"Distribution of {col}")
            elif col_type == "categorical":
                value_counts = df[col].value_counts().nlargest(10)
                sns.barplot(x=value_counts.values, y=value_counts.index)
                plt.title(f"Top 10 values in {col}")
            elif col_type == "datetime":
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    time_series = df[col].value_counts().sort_index()
                    sns.lineplot(x=time_series.index, y=time_series.values)
                    plt.title(f"Timeline of {col}")
            
            viz_path = os.path.join(temp_dir, f"{col}_plot.png")
            plt.savefig(viz_path, bbox_inches='tight')
            plt.close()
            viz_files.append(viz_path)
        except Exception as e:
            print(f"Failed to generate visualization for {col}: {str(e)}")
    
    return viz_files

def generate_excel_report(
    df: pd.DataFrame,
    analysis: Dict[str, Any],
    viz_files: list,
    output_path: str
):
    """Generate comprehensive Excel report"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Raw Data sheet
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        
        # Descriptive Stats sheet
        pd.DataFrame.from_dict(analysis["description"]).to_excel(
            writer, sheet_name="Descriptive Stats"
        )
        
        # Missing Values sheet
        pd.DataFrame.from_dict(
            {"column": list(analysis["missing_values"].keys()),
             "missing_count": list(analysis["missing_values"].values())
            }
        ).to_excel(writer, sheet_name="Missing Values", index=False)
        
        # Correlation sheet (if numeric columns exist)
        if analysis["correlation"]:
            pd.DataFrame.from_dict(analysis["correlation"]).to_excel(
                writer, sheet_name="Correlation Matrix"
            )
        
        # Visualizations sheet
        if viz_files:
            workbook = writer.book
            worksheet = workbook.create_sheet("Visualizations")
            
            row = 1
            for viz_path in viz_files:
                img = plt.imread(viz_path)
                img = Image.open(viz_path)
                img_width, img_height = img.size
                
                # Scale image to fit
                scale = 0.5
                from openpyxl.drawing.image import Image as XLImage
                img = XLImage(viz_path)
                worksheet.add_image(img, f"A{row}")
                row += int(img_height * scale / 15) + 2  # Add spacing