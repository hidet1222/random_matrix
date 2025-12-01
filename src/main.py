import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import value_and_grad, jit
import optax
import os

# --- 物理エンジン (完全自由度版: 16パラメータ) ---
def create_engine():
    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])

    def phase_shifter(voltage):
        # 簡易モデル: 電圧そのものを位相回転量とする (0 ~ 2pi)
        # 物理定数計算を入れると学習が不安定になることがあるため、
        # 「行列再現」の数学的証明では直接位相を動かします
        phi = voltage * jnp.pi 
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])

    def universal_mzi(v_theta, v_phi):
        # Clements型 MZI: 前段に位相シフタ(Phi) + MZI(Theta)
        # これで「どんな混合比」でも「どんな位相差」でも作れる最強の素子になる
        PS_phi = phase_shifter(v_phi)
        DC1 = directional_coupler()
        PS_theta = phase_shifter(v_theta)
        DC2 = directional_coupler()
        
        # Phi -> DC -> Theta -> DC
        return jnp.dot(DC2, jnp.dot(PS_theta, jnp.dot(DC1, PS_phi)))

    @jit
    def simulate_mesh(params):
        # パラメータ内訳 (Total 16)
        # 0~5:   MZIの混合比 (Theta)
        # 6~11:  MZIの入力位相 (Phi)
        # 12~15: 出力位相 (Output Phase)
        
        thetas = params[0:6]
        phis   = params[6:12]
        outs   = params[12:16]

        # 6-MZI Mesh (Universal Layout)
        # 各MZIに (theta, phi) の2つを与える
        T0 = universal_mzi(thetas[0], phis[0])
        T1 = universal_mzi(thetas[1], phis[1])
        L1 = jnp.block([[T0, jnp.zeros((2,2))], [jnp.zeros((2,2)), T1]])
        
        T2 = universal_mzi(thetas[2], phis[2])
        L2 = jnp.eye(4, dtype=complex)
        L2 = L2.at[1:3, 1:3].set(T2)

        T3 = universal_mzi(thetas[3], phis[3])
        T4 = universal_mzi(thetas[4], phis[4])
        L3 = jnp.block([[T3, jnp.zeros((2,2))], [jnp.zeros((2,2)), T4]])
        
        T5 = universal_mzi(thetas[5], phis[5])
        L4 = jnp.eye(4, dtype=complex)
        L4 = L4.at[1:3, 1:3].set(T5)
        
        U_mesh = jnp.dot(L4, jnp.dot(L3, jnp.dot(L2, L1)))
        
        # 最後の出力位相調整
        phase_matrix = jnp.diag(jnp.exp(1j * outs * jnp.pi))
        
        U_total = jnp.dot(phase_matrix, U_mesh)
        return U_total

    return simulate_mesh

def run_matrix_reproduction():
    print("🚀 DiffPhoton: 16-DOF Perfect Matrix Reproduction...")
    print("   Goal: Eliminate the residual loss by adding degrees of freedom.")

    mesh_fn = create_engine()

    # --- ターゲット作成 ---
    key = jax.random.PRNGKey(999)
    # ランダム行列生成
    random_mat = jax.random.normal(key, (4, 4)) + 1j * jax.random.normal(key, (4, 4))
    target_U, _ = jnp.linalg.qr(random_mat)
    
    print("   Target Matrix generated.")

    # --- 学習設定 ---
    @jit
    def loss_fn(params):
        current_U = mesh_fn(params)
        # 行列間距離
        diff = current_U - target_U
        loss = jnp.sum(jnp.abs(diff)**2)
        return loss

    # ★パラメータ数を 16 に設定 (4x4行列の自由度と一致)
    params = jax.random.uniform(key, shape=(16,), minval=-1.0, maxval=1.0)
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    loss_history = []
    print("   Training...", end="", flush=True)
    
    # じっくり3000回
    for i in range(3000):
        val, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(val)
        if i % 500 == 0: print(f"[{val:.4f}]", end="", flush=True)
        
    final_loss = loss_history[-1]
    final_U = mesh_fn(params)
    print(" Done!")
    print(f"   Final Matrix Distance (Loss): {final_loss:.8f}")

    # --- 結果保存 ---
    if not os.path.exists('output'): os.makedirs('output')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(jnp.abs(target_U), cmap='viridis', vmin=0, vmax=0.8)
    axes[0].set_title("Target")
    im2 = axes[1].imshow(jnp.abs(final_U), cmap='viridis', vmin=0, vmax=0.8)
    axes[1].set_title(f"Reproduced (Loss={final_loss:.1e})")
    
    plt.tight_layout()
    output_path = "output/matrix_reproduction.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Check Complete.")
    print(f"   Image saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_matrix_reproduction()