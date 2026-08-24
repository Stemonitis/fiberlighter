def pca_correction(signal_gcamp, signal_405, n_components=2):
    """
    Use PCA to separate artifacts from neural signal
    
    Idea: Stack both signals and find principal components
    Component 1 = shared artifacts (bleaching, motion)
    Component 2 = neural-specific signal
    """
    
    # Stack signals into matrix (samples × features)
    # Each row is a timepoint, each column is a signal
    data = np.column_stack([signal_405, signal_gcamp])
    print(f"Data shape: {data.shape} (samples × signals)")
    
    # Standardize (important for PCA!)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data_scaled)
    
    print(f"\nExplained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.1f}%")
    
    # Component 1 usually = artifacts (shared between both)
    # Component 2 usually = neural signal (GCaMP-specific)
    
    # Reconstruct WITHOUT component 1 (removes artifacts)
    components_cleaned = components.copy()
    components_cleaned[:, 0] = 0  # Zero out PC1 (artifacts)
    
    # Transform back to original space
    data_cleaned = pca.inverse_transform(components_cleaned)
    data_cleaned = scaler.inverse_transform(data_cleaned)
    
    # Extract cleaned GCaMP (column 1)
    gcamp_cleaned = data_cleaned[:, 1]
    
    return {
        'components': components,
        'explained_variance': pca.explained_variance_ratio_,
        'gcamp_cleaned': gcamp_cleaned,
        'pca_model': pca,
        'scaler': scaler
    }
def ica_correction(signal_gcamp, signal_405, n_components=3, random_state=42):
    
    # Stack signals
    data = np.column_stack([signal_405, signal_gcamp])
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
    sources = ica.fit_transform(data_scaled)
    
    print(f"ICA found {n_components} independent sources")
    
    # Plot all sources to identify which is which
    fig, axes = plt.subplots(n_components, 1, figsize=(14, 3*n_components))
    
    for i in range(n_components):
        axes[i].plot(time_aligned, sources[:, i])
        axes[i].set_title(f'Independent Component {i+1}')
        axes[i].set_ylabel(f'IC{i+1}')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('ica_sources.png', dpi=300)
    plt.show()
    
    # Now YOU identify which component is neural signal by looking at plots
    print("\nLook at the plots above:")
    print("- Slow drift = bleaching (remove)")
    print("- Fast spikes = motion (remove)")
    print("- Smooth transients = neural signal (KEEP!)")
    
    return {
        'sources': sources,
        'ica_model': ica,
        'scaler': scaler
    }