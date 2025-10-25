"""
Data transformation tasks using Prefect with comprehensive EDA capabilities.
Implements Sub-Objective 1.4: Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
from prefect import task
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@task(name="exploratory_data_analysis")
def exploratory_data_analysis(
    df: pd.DataFrame,
    target_column: str = "Churn",
    output_dir: str = "data/eda_outputs"
) -> Dict[str, Any]:
    """
    Comprehensive Exploratory Data Analysis for Telco Customer Churn dataset.
    
    Sub-Objective 1.4: EDA with correlation analysis and data visualization
    
    Args:
        df: Input dataframe
        target_column: Target variable for analysis
        output_dir: Directory to save EDA outputs and visualizations
    
    Returns:
        Dict containing all EDA results and statistics
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("🔍 Starting comprehensive Exploratory Data Analysis")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        eda_results = {
            'dataset_overview': {},
            'correlation_analysis': {},
            'univariate_analysis': {},
            'bivariate_analysis': {},
            'visualizations_saved': []
        }
        
        # 1. Dataset Overview
        eda_results['dataset_overview'] = get_dataset_overview(df, target_column)
        
        # 2. Correlation Analysis (numeric and categorical)
        eda_results['correlation_analysis'] = analyze_correlations(df, target_column, output_dir)
        
        # 3. Univariate Analysis with Visualizations
        eda_results['univariate_analysis'] = univariate_analysis(df, target_column, output_dir)
        
        # 4. Bivariate Analysis with Visualizations
        eda_results['bivariate_analysis'] = bivariate_analysis(df, target_column, output_dir)
        
        # 5. Feature Importance Analysis
        eda_results['feature_importance'] = analyze_feature_importance(df, target_column, output_dir)
        
        # 6. Save comprehensive EDA report
        save_eda_report(eda_results, output_dir)
        
        logger.info("✅ Comprehensive EDA completed successfully")
        return eda_results
        
    except Exception as e:
        logger.error(f"Error in exploratory data analysis: {str(e)}")
        raise


def get_dataset_overview(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Get comprehensive dataset overview and basic statistics"""
    
    overview = {
        'shape': df.shape,
        'total_features': df.shape[1],
        'total_records': df.shape[0],
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
        'data_types': {str(dtype): count for dtype, count in df.dtypes.value_counts().items()},
        'missing_values': {str(col): int(val) for col, val in df.isnull().sum().items()},
        'missing_percentage': {str(col): float(val) for col, val in (df.isnull().sum() / len(df) * 100).round(2).items()},
        'duplicate_records': int(df.duplicated().sum()),
        'target_distribution': {str(k): int(v) for k, v in df[target_column].value_counts().items()} if target_column in df.columns else {}
    }
    
    # Separate numeric and categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    overview['feature_types'] = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'numeric_count': len(numeric_features),
        'categorical_count': len(categorical_features)
    }
    
    # Basic statistics for numeric features
    if numeric_features:
        desc_stats = df[numeric_features].describe()
        overview['numeric_statistics'] = {
            str(col): {str(stat): float(val) for stat, val in col_stats.items()}
            for col, col_stats in desc_stats.to_dict().items()
        }
    
    # Categorical features statistics
    if categorical_features:
        cat_stats = {}
        for col in categorical_features:
            cat_stats[str(col)] = {
                'unique_values': int(df[col].nunique()),
                'unique_categories': [str(x) for x in df[col].unique().tolist()],
                'most_frequent': str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                'value_counts': {str(k): int(v) for k, v in df[col].value_counts().items()}
            }
        overview['categorical_statistics'] = cat_stats
    
    return overview


def analyze_correlations(df: pd.DataFrame, target_column: str, output_dir: str) -> Dict[str, Any]:
    """
    Analyze correlations between features:
    - Numeric-Numeric correlations (Pearson, Spearman)
    - Categorical-Categorical correlations (Cramér's V)
    - Numeric-Categorical correlations (Point-biserial, ANOVA)
    """
    
    correlation_results = {}
    
    # Separate feature types
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # 1. Numeric-Numeric Correlations
    if len(numeric_features) > 1:
        # Pearson correlation
        pearson_corr = df[numeric_features].corr(method='pearson')
        
        # Spearman correlation (for non-linear relationships)
        spearman_corr = df[numeric_features].corr(method='spearman')
        
        correlation_results['numeric_correlations'] = {
            'pearson_matrix': pearson_corr.to_dict(),
            'spearman_matrix': spearman_corr.to_dict(),
            'high_correlations_pearson': find_high_correlations(pearson_corr),
            'high_correlations_spearman': find_high_correlations(spearman_corr)
        }
        
        # Visualize numeric correlations
        create_correlation_heatmaps(pearson_corr, spearman_corr, output_dir)
    
    # 2. Categorical-Categorical Correlations (Cramér's V)
    if len(categorical_features) > 1:
        cramers_v_matrix = calculate_cramers_v_matrix(df, categorical_features)
        correlation_results['categorical_correlations'] = {
            'cramers_v_matrix': cramers_v_matrix,
            'high_associations': find_high_correlations(pd.DataFrame(cramers_v_matrix))
        }
        
        # Visualize categorical associations
        create_cramers_v_heatmap(cramers_v_matrix, output_dir)
    
    # 3. Target Variable Correlations
    if target_column in df.columns:
        target_correlations = analyze_target_correlations(df, target_column, numeric_features, categorical_features)
        correlation_results['target_correlations'] = target_correlations
        
        # Visualize target correlations
        create_target_correlation_plots(df, target_column, target_correlations, output_dir)
    
    return correlation_results


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
    """Find feature pairs with high correlation"""
    high_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value >= threshold:
                high_corr.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': round(corr_matrix.iloc[i, j], 3),
                    'abs_correlation': round(corr_value, 3)
                })
    
    return sorted(high_corr, key=lambda x: x['abs_correlation'], reverse=True)


def calculate_cramers_v_matrix(df: pd.DataFrame, categorical_features: List[str]) -> Dict:
    """Calculate Cramér's V for categorical-categorical associations"""
    n_features = len(categorical_features)
    cramers_matrix = np.zeros((n_features, n_features))
    
    for i, feat1 in enumerate(categorical_features):
        for j, feat2 in enumerate(categorical_features):
            if i == j:
                cramers_matrix[i, j] = 1.0
            else:
                cramers_matrix[i, j] = calculate_cramers_v(df[feat1], df[feat2])
    
    # Convert to dictionary format
    result = {}
    for i, feat1 in enumerate(categorical_features):
        result[feat1] = {}
        for j, feat2 in enumerate(categorical_features):
            result[feat1][feat2] = round(cramers_matrix[i, j], 3)
    
    return result


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V statistic for categorical association"""
    try:
        confusion_matrix = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape[0] - 1, confusion_matrix.shape[1] - 1)
        if min_dim == 0:
            return 0.0
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return cramers_v
    except:
        return 0.0


def analyze_target_correlations(df: pd.DataFrame, target_column: str, 
                               numeric_features: List[str], categorical_features: List[str]) -> Dict:
    """Analyze correlations between features and target variable"""
    
    target_corr = {}
    
    # For numeric features with target
    if target_column in df.select_dtypes(include=['object']).columns:
        # Target is categorical - use point-biserial correlation for numeric features
        target_encoded = pd.get_dummies(df[target_column]).iloc[:, 0]  # Binary encoding
        
        numeric_target_corr = {}
        for feature in numeric_features:
            if feature != target_column:
                try:
                    corr, p_value = pearsonr(df[feature].fillna(df[feature].mean()), target_encoded)
                    numeric_target_corr[feature] = {
                        'correlation': round(corr, 3),
                        'p_value': round(p_value, 4),
                        'significant': p_value < 0.05
                    }
                except:
                    numeric_target_corr[feature] = {'correlation': 0, 'p_value': 1, 'significant': False}
        
        target_corr['numeric_with_target'] = numeric_target_corr
    
    # For categorical features with target
    categorical_target_corr = {}
    for feature in categorical_features:
        if feature != target_column:
            cramers_v_value = calculate_cramers_v(df[feature], df[target_column])
            categorical_target_corr[feature] = {
                'cramers_v': round(cramers_v_value, 3),
                'association_strength': get_association_strength(cramers_v_value)
            }
    
    target_corr['categorical_with_target'] = categorical_target_corr
    
    return target_corr


def get_association_strength(cramers_v: float) -> str:
    """Categorize Cramér's V association strength"""
    if cramers_v < 0.1:
        return "negligible"
    elif cramers_v < 0.3:
        return "weak"
    elif cramers_v < 0.5:
        return "moderate"
    else:
        return "strong"


def univariate_analysis(df: pd.DataFrame, target_column: str, output_dir: str) -> Dict[str, Any]:
    """Perform univariate analysis with visualizations"""
    
    univariate_results = {}
    
    # Analyze numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_features:
        univariate_results['numeric'] = analyze_numeric_univariate(df, numeric_features, output_dir)
    
    if categorical_features:
        univariate_results['categorical'] = analyze_categorical_univariate(df, categorical_features, target_column, output_dir)
    
    return univariate_results


def analyze_numeric_univariate(df: pd.DataFrame, numeric_features: List[str], output_dir: str) -> Dict:
    """Analyze numeric features with histograms and box plots"""
    
    results = {}
    
    # Create subplots for numeric features
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Histograms
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, feature in enumerate(numeric_features):
        if i < len(axes):
            df[feature].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[i].legend()
            
            # Store statistics
            results[feature] = {
                'mean': round(mean_val, 2),
                'median': round(median_val, 2),
                'std': round(df[feature].std(), 2),
                'skewness': round(df[feature].skew(), 2),
                'kurtosis': round(df[feature].kurtosis(), 2),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'q25': df[feature].quantile(0.25),
                'q75': df[feature].quantile(0.75)
            }
    
    # Hide empty subplots
    for i in range(len(numeric_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/numeric_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Box plots
    if len(numeric_features) > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                df.boxplot(column=feature, ax=axes[i])
                axes[i].set_title(f'Box Plot of {feature}')
        
        # Hide empty subplots
        for i in range(len(numeric_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/numeric_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results


def analyze_categorical_univariate(df: pd.DataFrame, categorical_features: List[str], 
                                 target_column: str, output_dir: str) -> Dict:
    """Analyze categorical features with bar charts"""
    
    results = {}
    
    # Create bar charts for categorical features
    n_features = len(categorical_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, feature in enumerate(categorical_features):
        if i < len(axes):
            value_counts = df[feature].value_counts()
            value_counts.plot(kind='bar', ax=axes[i], rot=45)
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
            
            # Store statistics
            results[feature] = {
                'unique_count': df[feature].nunique(),
                'most_frequent': df[feature].mode().iloc[0] if not df[feature].mode().empty else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.idxmin() if len(value_counts) > 0 else None,
                'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.to_dict()
            }
    
    # Hide empty subplots
    for i in range(len(categorical_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/categorical_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def bivariate_analysis(df: pd.DataFrame, target_column: str, output_dir: str) -> Dict[str, Any]:
    """Perform bivariate analysis with visualizations"""
    
    bivariate_results = {}
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Numeric vs Target
    if numeric_features and target_column in df.columns:
        bivariate_results['numeric_vs_target'] = analyze_numeric_vs_target(df, numeric_features, target_column, output_dir)
    
    # Categorical vs Target
    if categorical_features and target_column in df.columns:
        bivariate_results['categorical_vs_target'] = analyze_categorical_vs_target(df, categorical_features, target_column, output_dir)
    
    # Numeric vs Numeric (scatter plots for key relationships)
    if len(numeric_features) >= 2:
        bivariate_results['numeric_vs_numeric'] = analyze_numeric_vs_numeric(df, numeric_features, output_dir)
    
    return bivariate_results


def analyze_numeric_vs_target(df: pd.DataFrame, numeric_features: List[str], 
                            target_column: str, output_dir: str) -> Dict:
    """Analyze numeric features against target with box plots and violin plots"""
    
    results = {}
    
    # Box plots of numeric features by target
    n_features = min(len(numeric_features), 6)  # Limit to 6 for readability
    selected_features = numeric_features[:n_features]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(selected_features):
        if i < len(axes):
            df.boxplot(column=feature, by=target_column, ax=axes[i])
            axes[i].set_title(f'{feature} by {target_column}')
            axes[i].set_xlabel(target_column)
            axes[i].set_ylabel(feature)
            
            # Calculate statistics by target
            grouped_stats = df.groupby(target_column)[feature].agg(['mean', 'median', 'std']).round(2)
            results[feature] = grouped_stats.to_dict()
    
    # Hide empty subplots
    for i in range(len(selected_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(f"{output_dir}/numeric_vs_target_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def analyze_categorical_vs_target(df: pd.DataFrame, categorical_features: List[str], 
                                target_column: str, output_dir: str) -> Dict:
    """Analyze categorical features against target with stacked bar charts"""
    
    results = {}
    
    # Remove target from categorical features if present
    features_to_plot = [f for f in categorical_features if f != target_column]
    n_features = min(len(features_to_plot), 6)
    selected_features = features_to_plot[:n_features]
    
    if selected_features:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(selected_features):
            if i < len(axes):
                # Create contingency table
                contingency = pd.crosstab(df[feature], df[target_column], normalize='index') * 100
                contingency.plot(kind='bar', stacked=True, ax=axes[i], rot=45)
                axes[i].set_title(f'{feature} vs {target_column} (Percentage)')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Percentage')
                axes[i].legend(title=target_column)
                
                # Store contingency table
                results[feature] = {
                    'contingency_table': pd.crosstab(df[feature], df[target_column]).to_dict(),
                    'percentage_by_category': contingency.to_dict()
                }
        
        # Hide empty subplots
        for i in range(len(selected_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/categorical_vs_target_stacked.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results


def analyze_numeric_vs_numeric(df: pd.DataFrame, numeric_features: List[str], output_dir: str) -> Dict:
    """Create scatter plots for key numeric feature relationships"""
    
    results = {}
    
    # Select top correlated pairs for scatter plots
    corr_matrix = df[numeric_features].corr()
    
    # Find top correlated pairs
    high_corr_pairs = []
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > 0.3:  # Only show moderately correlated features
                high_corr_pairs.append({
                    'feature1': numeric_features[i],
                    'feature2': numeric_features[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Plot top 6 pairs
    n_pairs = min(6, len(high_corr_pairs))
    if n_pairs > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, pair in enumerate(high_corr_pairs[:n_pairs]):
            if i < len(axes):
                x_feature = pair['feature1']
                y_feature = pair['feature2']
                
                axes[i].scatter(df[x_feature], df[y_feature], alpha=0.6)
                axes[i].set_xlabel(x_feature)
                axes[i].set_ylabel(y_feature)
                axes[i].set_title(f'{x_feature} vs {y_feature}\nCorr: {pair["correlation"]:.3f}')
                
                # Add trend line
                z = np.polyfit(df[x_feature].dropna(), df[y_feature].dropna(), 1)
                p = np.poly1d(z)
                axes[i].plot(df[x_feature], p(df[x_feature]), "r--", alpha=0.8)
                
                results[f"{x_feature}_vs_{y_feature}"] = pair
        
        # Hide empty subplots
        for i in range(n_pairs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/numeric_vs_numeric_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results


def analyze_feature_importance(df: pd.DataFrame, target_column: str, output_dir: str) -> Dict:
    """Analyze feature importance using mutual information"""
    
    try:
        # Prepare data for mutual information
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode categorical variables for mutual information
        X_encoded = pd.get_dummies(X, drop_first=True)
        y_encoded = pd.get_dummies(y).iloc[:, 0] if y.dtype == 'object' else y
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_encoded, y_encoded, random_state=42)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)  # Top 15 features
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mutual Information Score')
        plt.title(f'Top 15 Feature Importance (Mutual Information with {target_column})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'feature_importance_scores': feature_importance.to_dict('records'),
            'top_10_features': feature_importance.head(10)['feature'].tolist(),
            'method': 'mutual_information'
        }
        
    except Exception as e:
        return {'error': str(e)}


def create_correlation_heatmaps(pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame, output_dir: str):
    """Create correlation heatmaps"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pearson correlation heatmap
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, ax=ax1, fmt='.2f')
    ax1.set_title('Pearson Correlation Matrix')
    
    # Spearman correlation heatmap
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, ax=ax2, fmt='.2f')
    ax2.set_title('Spearman Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_cramers_v_heatmap(cramers_v_matrix: Dict, output_dir: str):
    """Create Cramér's V heatmap for categorical associations"""
    
    # Convert to DataFrame for plotting
    df_cramers = pd.DataFrame(cramers_v_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cramers, annot=True, cmap='viridis', fmt='.2f')
    plt.title("Cramér's V Association Matrix (Categorical Features)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cramers_v_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_target_correlation_plots(df: pd.DataFrame, target_column: str, 
                                  target_correlations: Dict, output_dir: str):
    """Create visualizations for target correlations"""
    
    # Plot numeric feature correlations with target
    if 'numeric_with_target' in target_correlations:
        numeric_corr = target_correlations['numeric_with_target']
        
        if numeric_corr:
            features = list(numeric_corr.keys())
            correlations = [numeric_corr[f]['correlation'] for f in features]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(features, correlations, color=['red' if c < 0 else 'blue' for c in correlations])
            plt.title(f'Numeric Features Correlation with {target_column}')
            plt.xlabel('Features')
            plt.ylabel('Correlation')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add correlation values on bars
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                        f'{corr:.2f}', ha='center', va='bottom' if height > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/numeric_target_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot categorical feature associations with target
    if 'categorical_with_target' in target_correlations:
        categorical_corr = target_correlations['categorical_with_target']
        
        if categorical_corr:
            features = list(categorical_corr.keys())
            cramers_values = [categorical_corr[f]['cramers_v'] for f in features]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(features, cramers_values, color='green', alpha=0.7)
            plt.title(f'Categorical Features Association with {target_column} (Cramér\'s V)')
            plt.xlabel('Features')
            plt.ylabel('Cramér\'s V')
            plt.xticks(rotation=45)
            
            # Add values on bars
            for bar, cv in zip(bars, cramers_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{cv:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/categorical_target_associations.png", dpi=300, bbox_inches='tight')
            plt.close()


def save_eda_report(eda_results: Dict, output_dir: str):
    """Save comprehensive EDA report as JSON"""
    
    import json
    
    # Convert numpy and pandas types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, pd.Int64Dtype, pd.Int32Dtype)):
            return int(obj) if not pd.isna(obj) else None
        elif isinstance(obj, (np.floating, pd.Float64Dtype, pd.Float32Dtype)):
            return float(obj) if not pd.isna(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'dtype'):  # Handle any pandas dtype objects
            return str(obj)
        elif isinstance(obj, dict):
            # Handle dictionary keys that might be pandas dtypes
            new_dict = {}
            for key, value in obj.items():
                # Convert keys to strings if they're not JSON serializable
                if hasattr(key, 'dtype') or hasattr(key, 'name'):
                    new_key = str(key)
                else:
                    new_key = key
                new_dict[new_key] = convert_numpy_types(value)
            return new_dict
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Handle other pandas objects
            return str(obj)
        else:
            return obj
    
    # Convert results with better error handling
    try:
        json_results = convert_numpy_types(eda_results)
        
        # Save to JSON file
        with open(f"{output_dir}/comprehensive_eda_report.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
            
    except (TypeError, ValueError) as e:
        # If JSON serialization still fails, save a simplified version
        print(f"Warning: Could not save full EDA report as JSON: {str(e)}")
        print("Saving simplified report instead...")
        
        # Create a simplified version that's guaranteed to be JSON serializable
        simplified_results = {
            'analysis_date': str(pd.Timestamp.now()),
            'dataset_shape': str(eda_results.get('dataset_overview', {}).get('shape', 'Unknown')),
            'feature_types': {
                'numeric_count': len(eda_results.get('dataset_overview', {}).get('feature_types', {}).get('numeric_features', [])),
                'categorical_count': len(eda_results.get('dataset_overview', {}).get('feature_types', {}).get('categorical_features', []))
            },
            'analysis_completed': True,
            'error_note': f"Full report could not be serialized: {str(e)}"
        }
        
        with open(f"{output_dir}/comprehensive_eda_report.json", 'w') as f:
            json.dump(simplified_results, f, indent=2)
    
    # Create summary report
    summary = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'dataset_shape': eda_results['dataset_overview']['shape'],
        'key_findings': {
            'high_correlations': len(eda_results.get('correlation_analysis', {}).get('numeric_correlations', {}).get('high_correlations_pearson', [])),
            'strong_categorical_associations': len([
                assoc for assoc in eda_results.get('correlation_analysis', {}).get('categorical_correlations', {}).get('high_associations', [])
                if assoc.get('abs_correlation', 0) > 0.5
            ]),
            'top_features_for_target': eda_results.get('feature_importance', {}).get('top_10_features', [])[:5]
        }
    }
    
    with open(f"{output_dir}/eda_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


# Keep existing tasks
@task(name="feature_engineering")
def feature_engineering(
    df: pd.DataFrame,
    feature_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply feature engineering transformations.
    
    Args:
        df: Input dataframe
        feature_config: Configuration for feature engineering
    
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting feature engineering")
        
        processed_df = df.copy()
        
        if feature_config is None:
            # Default feature engineering examples
            feature_config = {
                'polynomial_features': [],
                'interaction_features': [],
                'date_features': [],
                'binning_features': {},
                'log_transform': []
            }
        
        # Polynomial features
        for column in feature_config.get('polynomial_features', []):
            if column in processed_df.columns and processed_df[column].dtype in ['int64', 'float64']:
                processed_df[f"{column}_squared"] = processed_df[column] ** 2
                processed_df[f"{column}_cubed"] = processed_df[column] ** 3
                logger.info(f"Created polynomial features for {column}")
        
        # Interaction features
        for feature_pair in feature_config.get('interaction_features', []):
            if len(feature_pair) == 2 and all(col in processed_df.columns for col in feature_pair):
                col1, col2 = feature_pair
                if (processed_df[col1].dtype in ['int64', 'float64'] and 
                    processed_df[col2].dtype in ['int64', 'float64']):
                    processed_df[f"{col1}_{col2}_interaction"] = processed_df[col1] * processed_df[col2]
                    logger.info(f"Created interaction feature for {col1} and {col2}")
        
        # Date features
        for column in feature_config.get('date_features', []):
            if column in processed_df.columns:
                try:
                    processed_df[column] = pd.to_datetime(processed_df[column])
                    processed_df[f"{column}_year"] = processed_df[column].dt.year
                    processed_df[f"{column}_month"] = processed_df[column].dt.month
                    processed_df[f"{column}_day"] = processed_df[column].dt.day
                    processed_df[f"{column}_dayofweek"] = processed_df[column].dt.dayofweek
                    processed_df[f"{column}_quarter"] = processed_df[column].dt.quarter
                    logger.info(f"Created date features for {column}")
                except Exception as e:
                    logger.warning(f"Could not create date features for {column}: {e}")
        
        # Binning features
        for column, bins in feature_config.get('binning_features', {}).items():
            if column in processed_df.columns and processed_df[column].dtype in ['int64', 'float64']:
                processed_df[f"{column}_binned"] = pd.cut(processed_df[column], bins=bins, labels=False)
                logger.info(f"Created binned feature for {column}")
        
        # Log transformation
        for column in feature_config.get('log_transform', []):
            if column in processed_df.columns and processed_df[column].dtype in ['int64', 'float64']:
                # Add small constant to handle zeros
                processed_df[f"{column}_log"] = np.log1p(processed_df[column])
                logger.info(f"Created log-transformed feature for {column}")
        
        logger.info(f"Feature engineering completed. Shape: {df.shape} -> {processed_df.shape}")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


@task(name="split_train_test")
def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input dataframe
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        stratify: Whether to stratify the split based on target
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Splitting data into train and test sets")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise


@task(name="save_processed_data")
def save_processed_data(
    data: Dict[str, Any],
    output_dir: str,
    file_format: str = "csv"
) -> Dict[str, str]:
    """
    Save processed data to disk.
    
    Args:
        data: Dictionary containing data to save (e.g., X_train, X_test, y_train, y_test)
        output_dir: Directory to save data
        file_format: Format to save data ("csv", "parquet", "pickle")
    
    Returns:
        Dict[str, str]: Dictionary mapping data names to file paths
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Saving processed data")
        
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for data_name, data_obj in data.items():
            if data_obj is not None:
                if file_format == "csv":
                    file_path = Path(output_dir) / f"{data_name}.csv"
                    if isinstance(data_obj, pd.DataFrame):
                        data_obj.to_csv(file_path, index=False)
                    elif isinstance(data_obj, pd.Series):
                        data_obj.to_csv(file_path, header=True)
                    else:
                        # Convert to DataFrame if possible
                        pd.DataFrame(data_obj).to_csv(file_path, index=False)
                        
                elif file_format == "parquet":
                    file_path = Path(output_dir) / f"{data_name}.parquet"
                    if isinstance(data_obj, pd.DataFrame):
                        data_obj.to_parquet(file_path, index=False)
                    elif isinstance(data_obj, pd.Series):
                        data_obj.to_frame().to_parquet(file_path, index=False)
                    else:
                        pd.DataFrame(data_obj).to_parquet(file_path, index=False)
                        
                elif file_format == "pickle":
                    import pickle
                    file_path = Path(output_dir) / f"{data_name}.pkl"
                    with open(file_path, 'wb') as f:
                        pickle.dump(data_obj, f)
                
                saved_files[data_name] = str(file_path)
                logger.info(f"Saved {data_name} to {file_path}")
        
        logger.info(f"All processed data saved to {output_dir}")
        return saved_files
        
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise