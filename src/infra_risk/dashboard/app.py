"""
InfraRisk AI - Streamlit Dashboard
Interactive dashboard for infrastructure risk assessment
"""
import logging
from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

from src.infra_risk.schemas import (
    Project,
    CreditScore,
    DemandForecast,
    SiteProgress,
    ContagionIndex,
    CARULOutput,
    ContractRiskScore,
    CoverageRatios,
    SimulationState,
    SimulationResult,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="InfraRisk AI Dashboard",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    if "projects" not in st.session_state:
        st.session_state.projects = []
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = None
    
    # Sidebar
    st.sidebar.title("InfraRisk AI")
    st.sidebar.markdown("## Navigation")
    
    page = st.sidebar.selectbox(
        "Select Module",
        [
            "Dashboard Overview",
            "Project Portfolio",
            "Credit Scoring",
            "Demand Forecasting",
            "Geospatial Analysis",
            "GNN Contagion",
            "PINN CARUL",
            "Legal NLP",
            "Waterfall Analysis",
            "Simulation Lab",
        ],
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **InfraRisk AI**
        
        Multi-modal infrastructure project finance risk platform.
        
        - Credit Scoring Engine
        - Demand Forecasting (TFT)
        - Geospatial Progress Tracking
        - GNN Contagion Analysis
        - PINN Degradation Modeling
        - Legal NLP Pipeline
        - Cash Flow Waterfall
        - Simulation Lab
        """
    )
    
    # Render selected page
    if page == "Dashboard Overview":
        render_overview()
    elif page == "Project Portfolio":
        render_portfolio()
    elif page == "Credit Scoring":
        render_credit_scoring()
    elif page == "Demand Forecasting":
        render_demand_forecasting()
    elif page == "Geospatial Analysis":
        render_geospatial()
    elif page == "GNN Contagion":
        render_gnn()
    elif page == "PINN CARUL":
        render_pinn()
    elif page == "Legal NLP":
        render_legal()
    elif page == "Waterfall Analysis":
        render_waterfall()
    elif page == "Simulation Lab":
        render_simulation()


def render_overview():
    """Render dashboard overview"""
    st.title("🏗️ InfraRisk AI Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Projects", len(st.session_state.projects))
    with col2:
        st.metric("Portfolio Value", "$0M")
    with col3:
        st.metric("Avg DSCR", "N/A")
    with col4:
        st.metric("Avg PD", "N/A")
    with col5:
        st.metric("Active Simulations", 0)
    
    st.markdown("---")
    
    # Recent activity placeholder
    st.subheader("Recent Activity")
    st.info("No recent activity. Start by adding projects to your portfolio.")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add New Project"):
            st.session_state.selected_project = None
            st.rerun()
    with col2:
        if st.button("Run Credit Analysis"):
            st.info("Credit analysis requires a project to be selected.")
    with col3:
        if st.button("Launch Simulation"):
            st.info("Simulation Lab can be accessed from the navigation.")


def render_portfolio():
    """Render project portfolio management"""
    st.title("📁 Project Portfolio")
    
    # Add project form
    with st.expander("Add New Project", expanded=False):
        st.subheader("New Project Details")
        
        col1, col2 = st.columns(2)
        with col1:
            project_id = st.text_input("Project ID", value="PRJ-001")
            name = st.text_input("Project Name", value="New Infrastructure Project")
            sector = st.selectbox(
                "Sector",
                ["transport", "energy", "water", "telecom", "social", "industrial"],
            )
            sub_sector = st.text_input("Sub-Sector", value="toll_road")
            stage = st.selectbox(
                "Stage",
                ["concept", "pre_feasibility", "feasibility", "procurement",
                 "construction", "ramp_up", "operational", "refinancing"],
            )
        with col2:
            country = st.text_input("Country", value="Example Country")
            country_code = st.text_input("Country Code", value="EC")
            region = st.text_input("Region", value="Example Region")
            latitude = st.number_input("Latitude", value=0.0)
            longitude = st.number_input("Longitude", value=0.0)
            concession_years = st.number_input("Concession Years", value=25)
        
        col1, col2 = st.columns(2)
        with col1:
            total_capex = st.number_input("Total Capex (USD)", value=100_000_000)
            debt_amount = st.number_input("Debt Amount (USD)", value=70_000_000)
            equity_amount = st.number_input("Equity Amount (USD)", value=30_000_000)
        with col2:
            capacity = st.number_input("Capacity", value=100.0)
            capacity_unit = st.selectbox("Capacity Unit", ["MW", "km", "units", " ADT"])
            e_score = st.number_input("E-Score (0-100)", value=70.0)
            s_score = st.number_input("S-Score (0-100)", value=70.0)
            g_score = st.number_input("G-Score (0-100)", value=70.0)
        
        if st.button("Add Project"):
            project = Project(
                project_id=project_id,
                name=name,
                sector=sector,
                sub_sector=sub_sector,
                stage=stage,
                country=country,
                country_code=country_code,
                region=region,
                latitude=latitude,
                longitude=longitude,
                concession_years=concession_years,
                total_capex=total_capex,
                debt_amount=debt_amount,
                equity_amount=equity_amount,
                capacity=capacity,
                capacity_unit=capacity_unit,
                e_score=e_score,
                s_score=s_score,
                g_score=g_score,
                sponsors=[],
                lenders=[],
            )
            st.session_state.projects.append(project)
            st.success(f"Project {project_id} added to portfolio!")
            st.rerun()
    
    # Project list
    st.subheader("Project Portfolio")
    
    if not st.session_state.projects:
        st.info("No projects in portfolio. Add a project using the form above.")
        return
    
    # Project table
    project_data = []
    for p in st.session_state.projects:
        project_data.append({
            "ID": p.project_id,
            "Name": p.name,
            "Sector": p.sector,
            "Stage": p.stage,
            "Country": p.country,
            "Capex ($M)": round(p.total_capex / 1e6, 1),
            "Debt ($M)": round(p.debt_amount / 1e6, 1),
            "Equity ($M)": round(p.equity_amount / 1e6, 1),
        })
    
    df = pd.DataFrame(project_data)
    st.dataframe(df, use_container_width=True)
    
    # Project detail
    st.subheader("Project Details")
    selected_id = st.selectbox(
        "Select Project",
        [p.project_id for p in st.session_state.projects],
    )
    selected_project = next((p for p in st.session_state.projects if p.project_id == selected_id), None)
    
    if selected_project:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.json({
                "Project ID": selected_project.project_id,
                "Name": selected_project.name,
                "Sector": selected_project.sector,
                "Sub-Sector": selected_project.sub_sector,
                "Stage": selected_project.stage,
            })
        with col2:
            st.json({
                "Country": selected_project.country,
                "Region": selected_project.region,
                "Latitude": selected_project.latitude,
                "Longitude": selected_project.longitude,
                "Concession Years": selected_project.concession_years,
            })
        with col3:
            st.json({
                "Total Capex": f"${selected_project.total_capex:,.0f}",
                "Debt Amount": f"${selected_project.debt_amount:,.0f}",
                "Equity Amount": f"${selected_project.equity_amount:,.0f}",
                "Capacity": f"{selected_project.capacity} {selected_project.capacity_unit}",
            })
        
        st.markdown("---")
        st.subheader("Risk Scores")
        st.info("Run credit analysis to populate risk scores.")


def render_credit_scoring():
    """Render credit scoring analysis"""
    st.title("💳 Credit Scoring Analysis")
    
    st.info("Credit scoring analysis will be available once the credit ensemble model is trained.")
    
    # Placeholder for credit score display
    st.subheader("Credit Score Components")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probability of Default (1yr)", "N/A")
    with col2:
        st.metric("Loss Given Default", "N/A")
    with col3:
        st.metric("Expected Loss", "N/A")
    
    st.markdown("---")
    
    st.subheader("Key Risk Drivers")
    st.write("Top factors affecting credit risk:")
    st.write("1. DSCR levels")
    st.write("2. Contract risk score")
    st.write("3. Demand volatility")
    st.write("4. Contagion index")
    st.write("5. CARUL years")


def render_demand_forecasting():
    """Render demand forecasting analysis"""
    st.title("📈 Demand Forecasting")
    
    st.info("Demand forecasts are generated using Temporal Fusion Transformer (TFT).")
    
    # Placeholder
    st.subheader("Forecast Summary")
    st.write("Select a project to view demand forecast.")
    
    # Chart placeholder
    st.markdown("---")
    st.subheader("Demand Forecast Chart")
    st.write("Forecast visualization will appear here.")


def render_geospatial():
    """Render geospatial analysis"""
    st.title("🛰️ Geospatial Analysis")
    
    st.info("Geospatial progress tracking uses Sentinel-2 imagery and Siamese ResNet-50.")
    
    # Placeholder
    st.subheader("Construction Progress")
    st.write("Select a project to view progress tracking.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Progress", "N/A")
    with col2:
        st.metric("Confidence Interval", "N/A")
    
    st.markdown("---")
    st.subheader("Spectral Indices")
    st.write("NDVI, NDBI, NDWI analysis will appear here.")


def render_gnn():
    """Render GNN contagion analysis"""
    st.title("🕸️ GNN Contagion Analysis")
    
    st.info("Graph Neural Network analysis for systemic dependency mapping.")
    
    # Placeholder
    st.subheader("Contagion Index")
    st.write("Select a project to view contagion analysis.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Contagion Score", "N/A")
    with col2:
        st.metric("Systemic Importance", "N/A")
    with col3:
        st.metric("Portfolio Concentration", "N/A")
    
    st.markdown("---")
    st.subheader("Network Visualization")
    st.write("Dependency graph visualization will appear here.")


def render_pinn():
    """Render PINN CARUL analysis"""
    st.title("🔬 PINN CARUL Analysis")
    
    st.info("Physics-Informed Neural Network for Climate-Adjusted Remaining Useful Life.")
    
    # Placeholder
    st.subheader("Climate-Adjusted RUL")
    st.write("Select an asset to view CARUL analysis.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Base RUL (years)", "N/A")
    with col2:
        st.metric("CARUL (SSP2-4.5)", "N/A")
    with col3:
        st.metric("CARUL (SSP5-8.5)", "N/A")
    
    st.markdown("---")
    st.subheader("Degradation Trajectories")
    st.write("Degradation curves will appear here.")


def render_legal():
    """Render legal NLP analysis"""
    st.title("⚖️ Legal NLP Analysis")
    
    st.info("Contract intelligence using LayoutLM and Legal-BERT.")
    
    # Placeholder
    st.subheader("Contract Risk Score")
    st.write("Upload a contract document to analyze.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Risk Score", "N/A")
    with col2:
        st.metric("Covenant Strength", "N/A")
    
    st.markdown("---")
    st.subheader("Key Risk Flags")
    st.write("Risk flags will appear here after analysis.")


def render_waterfall():
    """Render waterfall analysis"""
    st.title("💰 Cash Flow Waterfall")
    
    st.info("Cash flow waterfall and coverage ratio analysis.")
    
    # Placeholder
    st.subheader("Coverage Ratios")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DSCR", "N/A")
    with col2:
        st.metric("LLCR", "N/A")
    with col3:
        st.metric("PLCR", "N/A")
    
    st.markdown("---")
    st.subheader("Cash Flow Components")
    st.write("Waterfall visualization will appear here.")


def render_simulation():
    """Render simulation lab"""
    st.title("🎮 Simulation Lab")
    
    st.info("Gamified infrastructure risk simulation with AI opponent.")
    
    # Game mode selection
    st.subheader("Select Game Mode")
    
    game_mode = st.selectbox(
        "Game Mode",
        [
            "Single Deal Tutorial",
            "Portfolio Manager",
            "Crisis Manager",
            "Deal Structurer",
        ],
    )
    
    st.markdown("---")
    
    # Game info
    game_info = {
        "Single Deal Tutorial": {
            "description": "Learn the basics with one project",
            "duration": "30 minutes",
            "difficulty": "Easy",
        },
        "Portfolio Manager": {
            "description": "Manage 10-15 deals across sectors",
            "duration": "90 minutes",
            "difficulty": "Medium",
        },
        "Crisis Manager": {
            "description": "Handle workouts and restructurings",
            "duration": "90 minutes",
            "difficulty": "Hard",
        },
        "Deal Structurer": {
            "description": "Structure greenfield projects from scratch",
            "duration": "90 minutes",
            "difficulty": "Expert",
        },
    }
    
    info = game_info.get(game_mode, {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Description:** {info.get('description', '')}")
    with col2:
        st.write(f"**Duration:** {info.get('duration', '')}")
    with col3:
        st.write(f"**Difficulty:** {info.get('difficulty', '')}")
    
    st.markdown("---")
    
    if st.button("Start Simulation", type="primary"):
        st.session_state.simulation_mode = game_mode
        st.success(f"Starting {game_mode}...")
        st.balloons()
    
    st.markdown("---")
    st.subheader("Simulation Controls")
    st.write("Simulation controls will appear here once started.")


if __name__ == "__main__":
    main()
