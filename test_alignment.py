import numpy as np
from dataset import SatelliteDatasetGenerator, extract_optical_features, extract_sar_features, extract_text_features
from model import CrossModalEmbeddingAligner

def run_gradient_check():
    """
    Performs numerical gradient checking to verify that the analytical backpropagation
    equations in model.py are mathematically correct.
    """
    print("Starting Gradient Check...")
    
    # 1. Generate small dataset
    generator = SatelliteDatasetGenerator(size=64)
    data = generator.generate_dataset(num_samples=8, seed=42)
    
    # 2. Extract features
    X_opt = np.array([extract_optical_features(p['optical']) for p in data])
    X_sar = np.array([extract_sar_features(p['sar']) for p in data])
    X_txt = np.array([extract_text_features(p['description'], generator.vocabulary) for p in data])
    
    # 3. Create Aligner
    aligner = CrossModalEmbeddingAligner(
        dim_opt=17, dim_sar=6, dim_txt=20,
        hidden_dim=8, embed_dim=4, temperature=0.5, seed=42
    )
    aligner.fit_normalizers(X_opt, X_sar, X_txt)
    
    # Normalize features
    X_opt_norm, X_sar_norm, X_txt_norm = aligner.normalize_features(X_opt, X_sar, X_txt)
    
    # Let's perform a step and get analytical gradients
    # We do a simplified forward pass and compute gradients
    z_opt, z_sar, z_txt = aligner.forward(X_opt_norm, X_sar_norm, X_txt_norm, normalized=False)
    
    u_opt = aligner.branch_opt.u
    u_sar = aligner.branch_sar.u
    u_txt = aligner.branch_txt.u
    
    # Compute losses and unnormalized grads
    loss_os, g_u_os_opt, g_u_os_sar = aligner.compute_contrastive_loss_and_grads(z_opt, z_sar, u_opt, u_sar)
    loss_ot, g_u_ot_opt, g_u_ot_txt = aligner.compute_contrastive_loss_and_grads(z_opt, z_txt, u_opt, u_txt)
    loss_st, g_u_st_sar, g_u_st_txt = aligner.compute_contrastive_loss_and_grads(z_sar, z_txt, u_sar, u_txt)
    
    total_loss = loss_os + loss_ot + loss_st
    
    g_u_opt = g_u_os_opt + g_u_ot_opt
    g_u_sar = g_u_os_sar + g_u_st_sar
    
    # Backward through SAR branch
    g_W1_sar, g_b1_sar, g_W2_sar, g_b2_sar = aligner.branch_sar.backward(g_u_sar)
    
    # Let's verify gradient of branch_sar.W2 (dimension: hidden_dim x embed_dim = 8 x 4)
    epsilon = 1e-4
    analytical_grads = g_W2_sar
    numerical_grads = np.zeros_like(analytical_grads)
    
    # Test a subset or all elements of W2
    for i in range(analytical_grads.shape[0]):
        for j in range(analytical_grads.shape[1]):
            # Perturb weight positively
            aligner.branch_sar.W2[i, j] += epsilon
            z_o, z_s, z_t = aligner.forward(X_opt_norm, X_sar_norm, X_txt_norm, normalized=False)
            u_o = aligner.branch_opt.u
            u_s = aligner.branch_sar.u
            u_t = aligner.branch_txt.u
            loss_os_p, _, _ = aligner.compute_contrastive_loss_and_grads(z_o, z_s, u_o, u_s)
            loss_ot_p, _, _ = aligner.compute_contrastive_loss_and_grads(z_o, z_t, u_o, u_t)
            loss_st_p, _, _ = aligner.compute_contrastive_loss_and_grads(z_s, z_t, u_s, u_t)
            loss_plus = loss_os_p + loss_ot_p + loss_st_p
            
            # Perturb weight negatively
            aligner.branch_sar.W2[i, j] -= 2.0 * epsilon
            z_o, z_s, z_t = aligner.forward(X_opt_norm, X_sar_norm, X_txt_norm, normalized=False)
            u_o = aligner.branch_opt.u
            u_s = aligner.branch_sar.u
            u_t = aligner.branch_txt.u
            loss_os_m, _, _ = aligner.compute_contrastive_loss_and_grads(z_o, z_s, u_o, u_s)
            loss_ot_m, _, _ = aligner.compute_contrastive_loss_and_grads(z_o, z_t, u_o, u_t)
            loss_st_m, _, _ = aligner.compute_contrastive_loss_and_grads(z_s, z_t, u_s, u_t)
            loss_minus = loss_os_m + loss_ot_m + loss_st_m
            
            # Reset weight
            aligner.branch_sar.W2[i, j] += epsilon
            
            # Numerical gradient
            numerical_grads[i, j] = (loss_plus - loss_minus) / (2.0 * epsilon)
            
    # Compute relative error
    numerator = np.linalg.norm(analytical_grads - numerical_grads)
    denominator = np.linalg.norm(analytical_grads) + np.linalg.norm(numerical_grads)
    rel_error = numerator / denominator
    
    print(f"Gradient check relative error for SAR branch Layer 2 weights: {rel_error:.2e}")
    if rel_error < 1e-4:
        print("GRADIENT CHECK PASSED! Backpropagation is mathematically correct.")
    else:
        print("GRADIENT CHECK FAILED! Please check backpropagation logic.")
        # Print some differences
        print("Analytical:\n", analytical_grads[:3, :3])
        print("Numerical:\n", numerical_grads[:3, :3])
        
    return rel_error < 1e-4

def test_model_training():
    """
    Verifies that training completes successfully and loss decreases.
    """
    print("\nTesting Model Training...")
    generator = SatelliteDatasetGenerator(size=64)
    data = generator.generate_dataset(num_samples=40, seed=42)
    
    X_opt = np.array([extract_optical_features(p['optical']) for p in data])
    X_sar = np.array([extract_sar_features(p['sar']) for p in data])
    X_txt = np.array([extract_text_features(p['description'], generator.vocabulary) for p in data])
    
    aligner = CrossModalEmbeddingAligner(
        dim_opt=17, dim_sar=6, dim_txt=20,
        hidden_dim=32, embed_dim=16, temperature=0.1, seed=42
    )
    aligner.fit_normalizers(X_opt, X_sar, X_txt)
    
    losses = []
    for epoch in range(15):
        metrics = aligner.train_step(X_opt, X_sar, X_txt, lr=0.01)
        losses.append(metrics['loss'])
        if epoch % 5 == 0 or epoch == 14:
            print(f"Epoch {epoch:02d} | Loss: {metrics['loss']:.4f} (OS: {metrics['loss_os']:.4f}, OT: {metrics['loss_ot']:.4f}, ST: {metrics['loss_st']:.4f})")
            
    # The loss should decrease or stay stable and not be NaN
    assert not np.isnan(losses[-1]), "Loss is NaN!"
    assert losses[-1] < losses[0], "Loss did not decrease!"
    print("Training test passed successfully.")

if __name__ == "__main__":
    grad_ok = run_gradient_check()
    test_model_training()
