"""
Vectory Theme System
Configurable themes for different preferences and accessibility needs.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class Theme:
    """Theme configuration dataclass."""
    name: str
    display_name: str
    description: str

    # Primary colors
    primary: str
    primary_dark: str
    secondary: str
    accent: str

    # Background colors
    bg_primary: str
    bg_secondary: str
    bg_card: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_muted: str

    # Border colors
    border: str
    border_hover: str

    # Status colors
    success: str
    warning: str
    error: str
    info: str

    # Gradient colors
    gradient_start: str
    gradient_mid: str
    gradient_end: str

    # Special colors
    terminal: str  # For terminal-style elements
    code_bg: str   # For code blocks

    # Font
    font_mono: str = "'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    font_sans: str = "'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


# Dark Developer Theme (Default)
DARK_THEME = Theme(
    name="dark",
    display_name="Dark Developer",
    description="Modern dark theme optimized for developers",

    primary="#6366f1",       # Indigo-500
    primary_dark="#4f46e5",  # Indigo-600
    secondary="#a855f7",     # Purple-500
    accent="#22d3ee",        # Cyan-400

    bg_primary="#0f172a",    # Slate-900
    bg_secondary="#1e293b",  # Slate-800
    bg_card="#1e293b",       # Slate-800

    text_primary="#f1f5f9",  # Slate-100
    text_secondary="#e2e8f0", # Slate-200
    text_muted="#94a3b8",    # Slate-400

    border="#334155",        # Slate-700
    border_hover="#475569",  # Slate-600

    success="#4ade80",       # Green-400
    warning="#fbbf24",       # Amber-400
    error="#f87171",         # Red-400
    info="#38bdf8",          # Sky-400

    gradient_start="#6366f1",
    gradient_mid="#8b5cf6",
    gradient_end="#a855f7",

    terminal="#22c55e",      # Green-500
    code_bg="#0f172a",
)


# Light Theme
LIGHT_THEME = Theme(
    name="light",
    display_name="Light",
    description="Clean light theme for bright environments",

    primary="#4f46e5",       # Indigo-600
    primary_dark="#4338ca",  # Indigo-700
    secondary="#7c3aed",     # Violet-600
    accent="#0891b2",        # Cyan-600

    bg_primary="#ffffff",    # White
    bg_secondary="#f8fafc",  # Slate-50
    bg_card="#ffffff",       # White

    text_primary="#0f172a",  # Slate-900
    text_secondary="#1e293b", # Slate-800
    text_muted="#64748b",    # Slate-500

    border="#e2e8f0",        # Slate-200
    border_hover="#cbd5e1",  # Slate-300

    success="#16a34a",       # Green-600
    warning="#d97706",       # Amber-600
    error="#dc2626",         # Red-600
    info="#0284c7",          # Sky-600

    gradient_start="#4f46e5",
    gradient_mid="#7c3aed",
    gradient_end="#a855f7",

    terminal="#16a34a",      # Green-600
    code_bg="#f1f5f9",       # Slate-100
)


# Colorblind-Friendly Theme (Deuteranopia/Protanopia safe)
# Uses blue-orange palette which is distinguishable for most types of color blindness
COLORBLIND_THEME = Theme(
    name="colorblind",
    display_name="High Contrast (Colorblind Safe)",
    description="Accessible theme using colorblind-friendly palette",

    primary="#0077bb",       # Blue (safe)
    primary_dark="#005588",  # Darker blue
    secondary="#ee7733",     # Orange (safe)
    accent="#33bbee",        # Cyan (safe)

    bg_primary="#1a1a2e",    # Dark blue-gray
    bg_secondary="#16213e",  # Slightly lighter
    bg_card="#1f2940",       # Card background

    text_primary="#ffffff",  # White
    text_secondary="#e8e8e8", # Light gray
    text_muted="#b0b0b0",    # Medium gray

    border="#404060",        # Blue-gray border
    border_hover="#505080",  # Lighter on hover

    success="#009988",       # Teal (distinguishable from red)
    warning="#ee7733",       # Orange
    error="#cc3311",         # Red-orange (more distinguishable)
    info="#0077bb",          # Blue

    gradient_start="#0077bb",
    gradient_mid="#33bbee",
    gradient_end="#009988",

    terminal="#33bbee",      # Cyan
    code_bg="#1a1a2e",
)


# High Contrast Dark Theme (for low vision users)
HIGH_CONTRAST_THEME = Theme(
    name="high_contrast",
    display_name="High Contrast Dark",
    description="Maximum contrast for low vision accessibility",

    primary="#ffff00",       # Yellow (high visibility)
    primary_dark="#cccc00",
    secondary="#00ffff",     # Cyan
    accent="#ff00ff",        # Magenta

    bg_primary="#000000",    # Pure black
    bg_secondary="#1a1a1a",  # Near black
    bg_card="#1a1a1a",

    text_primary="#ffffff",  # Pure white
    text_secondary="#ffffff",
    text_muted="#cccccc",

    border="#ffffff",        # White borders
    border_hover="#ffff00",  # Yellow on hover

    success="#00ff00",       # Bright green
    warning="#ffff00",       # Bright yellow
    error="#ff0000",         # Bright red
    info="#00ffff",          # Bright cyan

    gradient_start="#ffff00",
    gradient_mid="#00ffff",
    gradient_end="#ff00ff",

    terminal="#00ff00",      # Bright green
    code_bg="#000000",
)


# All available themes
THEMES: Dict[str, Theme] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "colorblind": COLORBLIND_THEME,
    "high_contrast": HIGH_CONTRAST_THEME,
}

DEFAULT_THEME = "dark"


def get_theme(theme_name: str) -> Theme:
    """Get a theme by name, falling back to default if not found."""
    return THEMES.get(theme_name, THEMES[DEFAULT_THEME])


def get_theme_css(theme: Theme) -> str:
    """Generate CSS variables for a theme."""
    return f"""
    <style>
    /* === Import Fonts with swap to prevent FOIT === */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

    /* === CRITICAL: Prevent all font/layout shifting === */
    * {{
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* === Theme Variables: {theme.display_name} === */
    :root {{
        --theme-primary: {theme.primary};
        --theme-primary-dark: {theme.primary_dark};
        --theme-secondary: {theme.secondary};
        --theme-accent: {theme.accent};

        --theme-bg-primary: {theme.bg_primary};
        --theme-bg-secondary: {theme.bg_secondary};
        --theme-bg-card: {theme.bg_card};

        --theme-text-primary: {theme.text_primary};
        --theme-text-secondary: {theme.text_secondary};
        --theme-text-muted: {theme.text_muted};

        --theme-border: {theme.border};
        --theme-border-hover: {theme.border_hover};

        --theme-success: {theme.success};
        --theme-warning: {theme.warning};
        --theme-error: {theme.error};
        --theme-info: {theme.info};

        --theme-gradient-start: {theme.gradient_start};
        --theme-gradient-mid: {theme.gradient_mid};
        --theme-gradient-end: {theme.gradient_end};

        --theme-terminal: {theme.terminal};
        --theme-code-bg: {theme.code_bg};

        --theme-font-mono: {theme.font_mono};
        --theme-font-sans: {theme.font_sans};
    }}

    /* === STREAMLIT OVERRIDES (Force Theme Colors) === */

    /* Main app background */
    .stApp {{
        background-color: {theme.bg_primary} !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {theme.bg_primary} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: {theme.bg_primary} !important;
    }}

    /* Main content area */
    .main .block-container {{
        background-color: {theme.bg_primary} !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {theme.bg_secondary} !important;
        background: linear-gradient(180deg, {theme.bg_secondary} 0%, {theme.bg_primary} 100%) !important;
        width: 300px !important;
        min-width: 300px !important;
    }}

    section[data-testid="stSidebar"] > div {{
        background-color: transparent !important;
        width: 300px !important;
    }}

    /* Sidebar content */
    [data-testid="stSidebarContent"] {{
        background-color: transparent !important;
        width: 100% !important;
    }}

    /* Smooth page transitions - prevent all jank */
    .stApp, .stApp * {{
        transition: none !important;
    }}

    .main .block-container {{
        transition: none !important;
        max-width: 100% !important;
    }}

    /* Prevent sidebar collapse animation jank */
    [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}

    /* Fixed font sizes to prevent text jumping */
    .stApp {{
        font-family: var(--theme-font-sans) !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }}

    /* Consistent content area */
    [data-testid="stAppViewBlockContainer"] {{
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    /* Prevent layout shift on page load */
    .element-container {{
        min-height: 0 !important;
    }}

    /* Text colors */
    .stApp, .stApp p, .stApp span, .stApp div {{
        color: {theme.text_primary};
    }}

    .stMarkdown, .stMarkdown p {{
        color: {theme.text_primary} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {theme.text_primary} !important;
    }}

    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {theme.text_primary} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {theme.primary} !important;
        color: white !important;
        border: none !important;
    }}

    .stButton > button:hover {{
        background-color: {theme.primary_dark} !important;
        border-color: {theme.primary_dark} !important;
    }}

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {{
        background-color: {theme.bg_card} !important;
        color: {theme.text_primary} !important;
        border-color: {theme.border} !important;
    }}

    /* Selectbox */
    [data-baseweb="select"] {{
        background-color: {theme.bg_card} !important;
    }}

    [data-baseweb="select"] > div {{
        background-color: {theme.bg_card} !important;
        border-color: {theme.border} !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {theme.bg_card} !important;
        color: {theme.text_primary} !important;
        border-color: {theme.border} !important;
    }}

    .streamlit-expanderContent {{
        background-color: {theme.bg_card} !important;
        border-color: {theme.border} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {theme.bg_secondary} !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {theme.text_muted} !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {theme.primary} !important;
    }}

    /* Dataframe */
    .stDataFrame {{
        background-color: {theme.bg_card} !important;
    }}

    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background-color: {theme.bg_card} !important;
    }}

    /* Alerts */
    .stAlert {{
        background-color: {theme.bg_card} !important;
        border-color: {theme.border} !important;
    }}

    /* Info/Success/Warning/Error boxes */
    [data-testid="stNotification"] {{
        background-color: {theme.bg_card} !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: {theme.bg_card} !important;
    }}

    [data-testid="stFileUploader"] section {{
        background-color: {theme.bg_card} !important;
        border-color: {theme.border} !important;
    }}

    /* Metric */
    [data-testid="stMetric"] {{
        background-color: {theme.bg_card} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {theme.primary} !important;
    }}

    /* Checkbox and Radio */
    .stCheckbox label,
    .stRadio label {{
        color: {theme.text_primary} !important;
    }}

    /* Slider */
    .stSlider label {{
        color: {theme.text_primary} !important;
    }}

    /* Multiselect */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {theme.primary} !important;
    }}

    /* Code blocks */
    .stCodeBlock {{
        background-color: {theme.code_bg} !important;
    }}

    code {{
        background-color: {theme.code_bg} !important;
        color: {theme.text_secondary} !important;
    }}

    /* Divider */
    hr {{
        border-color: {theme.border} !important;
    }}

    /* Links */
    a {{
        color: {theme.primary} !important;
    }}

    a:hover {{
        color: {theme.secondary} !important;
    }}

    /* === Global Animations === */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-30px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
    }}

    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 5px rgba({_hex_to_rgb(theme.primary)}, 0.3); }}
        50% {{ box-shadow: 0 0 20px rgba({_hex_to_rgb(theme.primary)}, 0.6); }}
    }}

    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}

    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    @keyframes terminalGlow {{
        0%, 100% {{ text-shadow: 0 0 5px rgba({_hex_to_rgb(theme.terminal)}, 0.5); }}
        50% {{ text-shadow: 0 0 20px rgba({_hex_to_rgb(theme.terminal)}, 0.8); }}
    }}

    /* === Hero Section === */
    .hero-title {{
        font-family: var(--theme-font-mono);
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, {theme.gradient_start} 0%, {theme.gradient_mid} 50%, {theme.gradient_end} 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 4s ease infinite;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }}

    .hero-subtitle {{
        font-family: var(--theme-font-mono);
        font-size: 1.1rem;
        color: {theme.text_muted};
        animation: fadeIn 1s ease-out 0.3s both;
    }}

    .hero-subtitle::before {{
        content: '> ';
        color: {theme.terminal};
    }}

    /* === Metric Cards === */
    .metric-card {{
        background: linear-gradient(135deg, {theme.bg_card} 0%, {theme.bg_primary} 100%);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out both;
        border: 1px solid {theme.border};
        font-family: var(--theme-font-mono);
    }}

    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba({_hex_to_rgb(theme.primary)}, 0.2);
        border-color: {theme.primary};
    }}

    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        font-family: var(--theme-font-mono);
        background: linear-gradient(135deg, {theme.primary}, {theme.secondary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .metric-label {{
        font-size: 0.85rem;
        color: {theme.text_muted};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 8px;
        font-family: var(--theme-font-mono);
    }}

    /* === Glass Cards === */
    .glass-card {{
        background: rgba({_hex_to_rgb(theme.bg_card)}, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 30px;
        border: 1px solid rgba({_hex_to_rgb(theme.primary)}, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.8s ease-out both;
    }}

    /* === Feature Cards === */
    .feature-card {{
        background: {theme.bg_card};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid {theme.border};
        animation: slideIn 0.5s ease-out both;
    }}

    .feature-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba({_hex_to_rgb(theme.primary)}, 0.2);
        border-color: {theme.primary};
    }}

    .feature-icon {{
        font-size: 2.5rem;
        margin-bottom: 16px;
        animation: float 3s ease-in-out infinite;
    }}

    .feature-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {theme.text_primary};
        margin-bottom: 8px;
        font-family: var(--theme-font-mono);
    }}

    .feature-desc {{
        font-size: 0.9rem;
        color: {theme.text_muted};
        line-height: 1.6;
    }}

    /* === Terminal Style === */
    .terminal-box {{
        background: {theme.code_bg};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 16px;
        font-family: var(--theme-font-mono);
        font-size: 0.9rem;
    }}

    .terminal-prompt {{
        color: {theme.terminal};
        animation: terminalGlow 2s ease-in-out infinite;
    }}

    .terminal-prompt::before {{
        content: '$ ';
        color: {theme.primary};
    }}

    /* === Code Block Style === */
    .code-block {{
        background: {theme.code_bg};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 16px 20px;
        font-family: var(--theme-font-mono);
        font-size: 0.85rem;
        color: {theme.text_secondary};
        overflow-x: auto;
    }}

    /* === Progress Bar Animation === */
    .animated-progress {{
        height: 6px;
        background: linear-gradient(90deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}, {theme.gradient_start});
        background-size: 300% 100%;
        border-radius: 3px;
        animation: gradient 2s linear infinite;
    }}

    /* === Glowing Button === */
    .glow-button {{
        background: linear-gradient(135deg, {theme.primary}, {theme.secondary});
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 8px;
        font-weight: 600;
        font-family: var(--theme-font-mono);
        cursor: pointer;
        transition: all 0.3s ease;
        animation: glow 2s ease-in-out infinite;
    }}

    .glow-button:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba({_hex_to_rgb(theme.primary)}, 0.5);
    }}

    /* === Animated List Items === */
    .animated-list-item {{
        animation: slideIn 0.5s ease-out both;
        padding: 16px;
        background: {theme.bg_card};
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 3px solid {theme.primary};
        transition: all 0.3s ease;
        font-family: var(--theme-font-mono);
    }}

    .animated-list-item:hover {{
        transform: translateX(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border-left-color: {theme.secondary};
    }}

    /* === Rank Badge === */
    .rank-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: var(--theme-font-mono);
        animation: pulse 2s ease-in-out infinite;
    }}

    .rank-1 {{ background: linear-gradient(135deg, #ffd700, #ffed4a); color: #0f172a; }}
    .rank-2 {{ background: linear-gradient(135deg, #94a3b8, #cbd5e1); color: #0f172a; }}
    .rank-3 {{ background: linear-gradient(135deg, #cd7f32, #d97706); color: #fff; }}

    /* === Shimmer Loading === */
    .shimmer {{
        background: linear-gradient(90deg, {theme.bg_card} 25%, {theme.border} 50%, {theme.bg_card} 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
    }}

    /* === Stats Counter === */
    .stats-counter {{
        font-size: 3rem;
        font-weight: 800;
        font-family: var(--theme-font-mono);
        background: linear-gradient(135deg, {theme.primary}, {theme.secondary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* === Animated Gradient Background === */
    .gradient-bg {{
        background: linear-gradient(-45deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}, {theme.primary_dark});
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        border-radius: 16px;
        padding: 40px;
        color: white;
    }}

    /* === Score Bar === */
    .score-bar-container {{
        background: {theme.border};
        border-radius: 6px;
        overflow: hidden;
        height: 10px;
    }}

    .score-bar {{
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, {theme.primary}, {theme.secondary});
        transition: width 1s ease-out;
    }}

    /* === Table Enhancements === */
    .stDataFrame {{
        animation: fadeIn 0.8s ease-out both;
    }}

    /* === Sidebar === */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {theme.bg_secondary} 0%, {theme.bg_primary} 100%);
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        color: {theme.text_primary};
    }}

    /* === Custom Scrollbar === */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {theme.bg_secondary};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {theme.primary}, {theme.secondary});
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {theme.primary_dark}, {theme.primary});
    }}

    /* === Vectory Logo Text === */
    .vectory-logo {{
        font-family: var(--theme-font-mono);
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end});
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 4s ease infinite;
        letter-spacing: -1px;
    }}

    .vectory-tagline {{
        font-family: var(--theme-font-mono);
        font-size: 0.75rem;
        color: {theme.text_muted};
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    /* === Page Header === */
    .page-header {{
        padding: 20px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid {theme.border};
    }}

    .page-header h1 {{
        font-family: var(--theme-font-mono);
        background: linear-gradient(135deg, {theme.primary}, {theme.secondary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }}

    /* === Status Badges === */
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: var(--theme-font-mono);
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .status-success {{ background: rgba({_hex_to_rgb(theme.success)}, 0.2); color: {theme.success}; border: 1px solid {theme.success}; }}
    .status-warning {{ background: rgba({_hex_to_rgb(theme.warning)}, 0.2); color: {theme.warning}; border: 1px solid {theme.warning}; }}
    .status-error {{ background: rgba({_hex_to_rgb(theme.error)}, 0.2); color: {theme.error}; border: 1px solid {theme.error}; }}
    .status-info {{ background: rgba({_hex_to_rgb(theme.info)}, 0.2); color: {theme.info}; border: 1px solid {theme.info}; }}

    /* === Delay Classes === */
    .delay-1 {{ animation-delay: 0.1s; }}
    .delay-2 {{ animation-delay: 0.2s; }}
    .delay-3 {{ animation-delay: 0.3s; }}
    .delay-4 {{ animation-delay: 0.4s; }}
    .delay-5 {{ animation-delay: 0.5s; }}

    /* === Dev Stat === */
    .dev-stat {{
        font-family: var(--theme-font-mono);
        background: {theme.bg_card};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 12px 16px;
    }}

    .dev-stat-label {{
        color: {theme.text_muted};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .dev-stat-value {{
        color: {theme.accent};
        font-size: 1.5rem;
        font-weight: 700;
    }}

    /* === Theme-specific page styles === */
    .themed-header {{
        font-family: var(--theme-font-mono);
        background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }}

    .themed-subtitle {{
        font-family: var(--theme-font-mono);
        color: {theme.text_muted};
        font-size: 1rem;
        margin-top: -10px;
    }}

    .themed-subtitle::before {{
        content: '$ ';
        color: {theme.terminal};
    }}

    .themed-banner {{
        background: linear-gradient(-45deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}, {theme.primary_dark});
        background-size: 400% 400%;
        animation: gradient 8s ease infinite;
        border-radius: 12px;
        padding: 30px;
        color: white;
        text-align: center;
    }}

    .themed-card {{
        background: {theme.bg_card};
        border: 1px solid {theme.border};
        border-radius: 12px;
        padding: 24px;
    }}

    .themed-link {{
        color: {theme.primary};
        text-decoration: none;
    }}

    .themed-link:hover {{
        color: {theme.secondary};
    }}

    /* === Hide Default Streamlit Elements === */
    /* Hide the default logo container */
    [data-testid="stLogo"] {{
        display: none !important;
    }}

    /* Hide the collapse button */
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}

    /* === Replace "app" with "⚡ Vectory" in sidebar nav === */
    /* Target the first nav item (app) and restyle it */
    [data-testid="stSidebarNav"] li:first-child a {{
        font-size: 0 !important;
        padding: 12px 16px !important;
    }}

    [data-testid="stSidebarNav"] li:first-child a::before {{
        content: "⚡ Vectory";
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: {theme.text_primary} !important;
        font-family: 'DM Sans', sans-serif !important;
    }}

    [data-testid="stSidebarNav"] li:first-child a span {{
        display: none !important;
    }}

    /* Keep the existing /VectoryBenchmark route but display the spaced product name. */
    [data-testid="stSidebarNav"] a[href$="VectoryBenchmark"] {{
        font-size: 0 !important;
    }}

    [data-testid="stSidebarNav"] a[href$="VectoryBenchmark"]::after {{
        content: "Vectory Benchmark";
        font-size: 14px !important;
        color: {theme.text_primary} !important;
        font-family: 'DM Sans', sans-serif !important;
    }}

    /* === Always Show Navigation (Remove Toggle Arrow) === */
    /* Remove collapse/expand functionality - always show nav */
    [data-testid="stSidebarNav"] {{
        overflow: visible !important;
    }}

    /* Hide the toggle/collapse button */
    [data-testid="stSidebarNavCollapseIcon"],
    [data-testid="collapsedControl"],
    button[kind="headerNoPadding"] {{
        display: none !important;
    }}

    /* Ensure nav items are always visible */
    [data-testid="stSidebarNavItems"] {{
        display: block !important;
        visibility: visible !important;
        max-height: none !important;
        overflow: visible !important;
    }}

    /* Remove the collapse animation */
    [data-testid="stSidebarNav"] ul {{
        max-height: none !important;
        overflow: visible !important;
    }}

    /* === Admin Section === */
    .admin-section {{
        position: absolute;
        bottom: 20px;
        left: 0;
        right: 0;
        padding: 16px;
        border-top: 1px solid {theme.border};
        background: linear-gradient(180deg, transparent 0%, rgba({_hex_to_rgb(theme.bg_primary)}, 0.95) 20%);
    }}

    .admin-section-title {{
        font-family: var(--theme-font-mono);
        font-size: 0.65rem;
        color: {theme.text_muted};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }}

    .admin-button {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        background: {theme.bg_card};
        border: 1px solid {theme.border};
        border-radius: 8px;
        color: {theme.text_secondary};
        font-family: var(--theme-font-mono);
        font-size: 0.85rem;
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
    }}

    .admin-button:hover {{
        background: {theme.primary};
        color: white;
        border-color: {theme.primary};
        transform: translateX(4px);
    }}

    .admin-button-icon {{
        font-size: 1rem;
    }}

    /* Adjust sidebar content to make room for admin section */
    [data-testid="stSidebarContent"] {{
        padding-bottom: 100px !important;
    }}

    /* === Hide "Made with Streamlit" Footer === */
    footer {{
        visibility: hidden !important;
    }}

    footer:after {{
        visibility: hidden !important;
    }}

    /* Hide the Streamlit menu hamburger footer text */
    .viewerBadge_container__r5tak {{
        display: none !important;
    }}

    /* Hide "Made with Streamlit" in main content */
    [data-testid="stToolbar"] {{
        display: none !important;
    }}

    /* === Animated Logo Container === */
    .animated-logo-container {{
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }}

    .animated-logo-container svg {{
        filter: drop-shadow(0 0 20px rgba({_hex_to_rgb(theme.primary)}, 0.3));
        transition: filter 0.3s ease;
    }}

    .animated-logo-container:hover svg {{
        filter: drop-shadow(0 0 30px rgba({_hex_to_rgb(theme.primary)}, 0.5));
    }}

    /* === Sidebar Navigation Enhancement === */
    [data-testid="stSidebarNav"] a {{
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: {theme.text_secondary} !important;
    }}

    [data-testid="stSidebarNav"] a:hover {{
        color: {theme.text_primary} !important;
    }}

    [data-testid="stSidebarNav"] a[aria-selected="true"] {{
        color: {theme.text_primary} !important;
        font-weight: 600 !important;
    }}

    /* === Header Styling === */
    .app-header {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(180deg, {theme.bg_primary} 0%, {theme.bg_primary}ee 80%, transparent 100%);
        padding: 16px 0;
        margin-bottom: 20px;
    }}

    /* === Feature Cards Enhancement === */
    .feature-card {{
        position: relative;
        overflow: hidden;
    }}

    .feature-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba({_hex_to_rgb(theme.primary)}, 0.1), transparent);
        transition: left 0.5s ease;
    }}

    .feature-card:hover::before {{
        left: 100%;
    }}

    /* === Vectory Workbench Design System === */
    .main .block-container {{
        max-width: 1260px !important;
        padding-top: 2.25rem !important;
        padding-left: 2.25rem !important;
        padding-right: 2.25rem !important;
    }}

    .stApp {{
        font-family: var(--theme-font-sans) !important;
        font-size: 15px !important;
        line-height: 1.55 !important;
        letter-spacing: 0 !important;
    }}

    h1, h2, h3, h4, h5, h6,
    .hero-title,
    .feature-title,
    .metric-value,
    .metric-label,
    .page-header h1,
    .section-title,
    .themed-header {{
        font-family: var(--theme-font-sans) !important;
        letter-spacing: 0 !important;
    }}

    .feature-card,
    .metric-card,
    .glass-card,
    .themed-card,
    .dev-stat {{
        border-radius: 8px !important;
        box-shadow: none !important;
        background: {theme.bg_card} !important;
        border: 1px solid {theme.border} !important;
    }}

    .feature-card {{
        min-height: 150px !important;
        padding: 18px !important;
        transition: border-color 160ms ease, background-color 160ms ease !important;
    }}

    .feature-card:hover,
    .metric-card:hover,
    .animated-list-item:hover {{
        transform: none !important;
        border-color: {theme.border_hover} !important;
        box-shadow: none !important;
    }}

    .feature-card::before,
    .feature-icon {{
        animation: none !important;
    }}

    .feature-icon {{
        font-size: 1.5rem !important;
        margin-bottom: 10px !important;
    }}

    .feature-title {{
        font-size: 1rem !important;
        font-weight: 700 !important;
        margin-bottom: 6px !important;
    }}

    .feature-desc {{
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }}

    .metric-card {{
        padding: 16px !important;
        min-height: 112px !important;
    }}

    .metric-value {{
        font-size: 1.8rem !important;
        font-weight: 750 !important;
    }}

    .metric-label,
    .dev-stat-label,
    .status-badge,
    .admin-section-title {{
        letter-spacing: 0.04em !important;
    }}

    .page-header {{
        padding: 0 0 16px 0 !important;
        margin-bottom: 20px !important;
    }}

    .page-header h1 {{
        font-size: clamp(1.8rem, 4vw, 2.5rem) !important;
        line-height: 1.1 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px !important;
        background: transparent !important;
        border-bottom: 1px solid {theme.border} !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 40px !important;
        padding: 0 12px !important;
        border-radius: 6px 6px 0 0 !important;
        font-weight: 600 !important;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        min-height: 40px !important;
        border-radius: 7px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }}

    [data-testid="stSidebar"] {{
        border-right: 1px solid {theme.border} !important;
    }}

    [data-testid="stSidebarNav"] a {{
        border-radius: 7px !important;
        min-height: 36px !important;
        margin: 2px 0 !important;
    }}

    [data-testid="stFileUploader"] section {{
        border-radius: 8px !important;
        border-style: dashed !important;
    }}

    @media (max-width: 900px) {{
        .main .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}

    </style>
    """


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB values for rgba() usage."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"
