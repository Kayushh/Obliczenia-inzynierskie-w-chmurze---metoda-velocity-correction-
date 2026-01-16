import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import os
import datetime

def make_mesh(x_lim, y_lim, num_elm_edge, meshplot):
    num_el = num_elm_edge**2
    n_x = num_elm_edge + 1
    n_y = num_elm_edge + 1
    num_nodes = n_x * n_y
    
    x_vals = np.linspace(x_lim[0], x_lim[1], n_x)
    y_vals = np.linspace(y_lim[0], y_lim[1], n_y)
    
    coords = np.zeros((num_nodes, 2))
    k = 0
    for j in range(n_y):
        for i in range(n_x):
            coords[k, 0] = x_vals[i]
            coords[k, 1] = y_vals[j]
            k += 1
            
    mesh_matrix = np.zeros((num_el, 5), dtype=int)
    # W Pythonie pierwsza kolumna to ID elementu, reszta to węzły
    mesh_matrix[:, 0] = np.arange(num_el)
    
    e = 0
    for j in range(n_y - 1):
        for i in range(n_x - 1):
            bottom_left_node = j * n_x + i
            nodeA = bottom_left_node
            nodeB = bottom_left_node + 1
            nodeC = bottom_left_node + n_x + 1
            nodeD = bottom_left_node + n_x
            mesh_matrix[e, 1:5] = [nodeA, nodeB, nodeC, nodeD]
            e += 1
            
    mesh_XY = np.zeros((num_el, 9))
    mesh_XY[:, 0] = np.arange(num_el)
    
    for e in range(num_el):
        nodes_for_element = mesh_matrix[e, 1:5]
        element_coords = coords[nodes_for_element, :]
        mesh_XY[e, 1:9] = element_coords.reshape(1, 8) 

    if meshplot == 1:
        plt.figure()
        plt.axis('equal')
        for i in range(num_el):
            node_indices = mesh_matrix[i, 1:5]
            nodes_plot = np.append(node_indices, node_indices[0])
            X = coords[nodes_plot, 0]
            Y = coords[nodes_plot, 1]
            plt.fill(X, Y, facecolor='w', edgecolor='b')
        
        plt.plot(coords[:, 0], coords[:, 1], 'r.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Mesh')
        plt.show()
        
    return coords, num_el, num_nodes, mesh_matrix, mesh_XY

def mass_matrix(num_nodes, num_el, mesh_matrix):
    h = 1.0 / np.sqrt(num_el)
    M_local = (h**2) * (1/36.0) * np.array([
        [4, 2, 1, 2],
        [2, 4, 2, 1],
        [1, 2, 4, 2],
        [2, 1, 2, 4]
    ])
    
    M_global = sp.lil_matrix((num_nodes, num_nodes))
    
    for e in range(num_el):
        node_index_global = mesh_matrix[e, 1:5]
        for i in range(4):
            for j in range(4):
                global_row = node_index_global[i]
                global_col = node_index_global[j]
                M_global[global_row, global_col] += M_local[i, j]
                
    return M_global.tocsr()

def stiffness_matrix(num_nodes, num_el, mesh_matrix):
    h = 1.0 / np.sqrt(num_el)
    K_local = (1.0 / (6.0 * h)) * np.array([
        [ 4, -1, -2, -1],
        [-1,  4, -1, -2],
        [-2, -1,  4, -1],
        [-1, -2, -1,  4]
    ])
    
    K_global = sp.lil_matrix((num_nodes, num_nodes))
    
    for e in range(num_el):
        node_index_global = mesh_matrix[e, 1:5]
        for i in range(4):
            for j in range(4):
                global_row = node_index_global[i]
                global_col = node_index_global[j]
                K_global[global_row, global_col] += K_local[i, j]
                
    return K_global.tocsr()

def div_matrix(num_nodes, num_el, mesh_matrix):
    h = 1.0 / np.sqrt(num_el)
    Rx_local = h * (1/12.0) * np.array([
        [-2,  2,  1, -1],
        [ 2, -2, -1,  1],
        [ 1, -1, -2,  2],
        [-1,  1,  2, -2]
    ])
    Ry_local = h * (1/12.0) * np.array([
        [-2, -1,  1,  2],
        [-1, -2,  2,  1],
        [ 1,  2, -2, -1],
        [ 2,  1, -1, -2]
    ])
    
    Rx_global = sp.lil_matrix((num_nodes, num_nodes))
    Ry_global = sp.lil_matrix((num_nodes, num_nodes))
    
    for e in range(num_el):
        node_index_global = mesh_matrix[e, 1:5]
        for i in range(4):
            for j in range(4):
                global_row = node_index_global[i]
                global_col = node_index_global[j]
                Rx_global[global_row, global_col] += Rx_local[i, j]
                Ry_global[global_row, global_col] += Ry_local[i, j]
                
    return Rx_global.tocsr(), Ry_global.tocsr()

def exact_solution(coords, t_end, deltat, nu):
    num_steps = int(np.round(t_end / deltat))
    num_nodes = coords.shape[0]
    
    F_exact = np.zeros((num_nodes, 2 * num_steps))
    u = np.zeros((num_nodes, num_steps))
    v = np.zeros((num_nodes, num_steps))
    p_exact = np.zeros((num_nodes, num_steps))
    v_exact = np.zeros((num_nodes, 2 * num_steps))
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    for i_step in range(num_steps):
        time_t = (i_step + 1) * deltat
        
        A_t = np.exp(-2 * time_t) * np.sin(np.pi * time_t)
        A_t_prim = np.exp(-2 * time_t) * (np.pi * np.cos(np.pi * time_t) - 2 * np.sin(np.pi * time_t))
        
        u[:, i_step] = A_t * (1 - np.cos(2 * np.pi * x)) * y * (2 - 3 * y)
        v[:, i_step] = -2 * np.pi * A_t * np.sin(2 * np.pi * x) * y**2 * (1 - y)
        
        # Wypełnianie v_exact [u, v, u, v...]
        v_exact[:, 2*i_step] = u[:, i_step]
        v_exact[:, 2*i_step + 1] = v[:, i_step]
        
        p_exact[:, i_step] = nu * ((x - 0.5)**2 + (y - 0.5)**2)
        
        du_dt = A_t_prim * (1 - np.cos(2 * np.pi * x)) * y * (2 - 3 * y)
        d2u_dx2 = A_t * (4 * np.pi**2 * np.cos(2 * np.pi * x)) * y * (2 - 3 * y)
        d2u_dy2 = A_t * (1 - np.cos(2 * np.pi * x)) * (-6)
        
        dv_dt = -A_t_prim * 2 * np.pi * np.sin(2 * np.pi * x) * y**2 * (1 - y)
        d2v_dx2 = A_t * 8 * np.pi**3 * np.sin(2 * np.pi * x) * y**2 * (1 - y)
        d2v_dy2 = -A_t * 2 * np.pi * np.sin(2 * np.pi * x) * (2 - 6 * y)
        
        dp_dx = 2 * nu * (x - 0.5)
        dp_dy = 2 * nu * (y - 0.5)
        
        laplacian_u = d2u_dx2 + d2u_dy2
        laplacian_v = d2v_dx2 + d2v_dy2
        
        fx_exact = du_dt - nu * laplacian_u + dp_dx
        fy_exact = dv_dt - nu * laplacian_v + dp_dy
        
        F_exact[:, 2*i_step] = fx_exact
        F_exact[:, 2*i_step + 1] = fy_exact
        
    return v_exact, p_exact, F_exact


def rhs_phi(f_kp1, f_k, u_wave_k, u_wave_km1, u_wave_km2, Rx_global, Ry_global, delta_t, num_nodes, is_startup):
    if is_startup:
        u_wave_diff = (u_wave_k - u_wave_km1) / delta_t
    else:
        u_wave_diff = (1.0 / (2.0 * delta_t)) * (7 * u_wave_k - 5 * u_wave_km1 + u_wave_km2)
        
    g = f_kp1 - f_k
    
    term1 = Rx_global.dot(g[:, 0])
    term2 = Ry_global.dot(g[:, 1])
    term3 = Rx_global.dot(u_wave_diff[:, 0])
    term4 = Ry_global.dot(u_wave_diff[:, 1])
    
    F_phi = term1 + term2 + term3 + term4
    return F_phi

def rhs_u_wave(f_kp1, u_wave_k, u_wave_km1, p_kp1, M_global, Rx_global, Ry_global, deltat, num_nodes, is_startup):
    F_wave = np.zeros((num_nodes, 2))
    g = f_kp1
    gx_nodes = g[:, 0]
    gy_nodes = g[:, 1]
    
    F_wave[:, 0] = M_global.dot(gx_nodes)
    F_wave[:, 1] = M_global.dot(gy_nodes)
    
    
    grad_p_x = Rx_global.transpose().dot(p_kp1)
    grad_p_y = Ry_global.transpose().dot(p_kp1)
    
    F_wave[:, 0] -= grad_p_x
    F_wave[:, 1] -= grad_p_y
    
    if is_startup:
        F_wave[:, 0] += (1.0 / deltat) * M_global.dot(u_wave_k[:, 0])
        F_wave[:, 1] += (1.0 / deltat) * M_global.dot(u_wave_k[:, 1])
    else:
        term_x = 4 * u_wave_k[:, 0] - u_wave_km1[:, 0]
        term_y = 4 * u_wave_k[:, 1] - u_wave_km1[:, 1]
        F_wave[:, 0] += (1.0 / (2 * deltat)) * M_global.dot(term_x)
        F_wave[:, 1] += (1.0 / (2 * deltat)) * M_global.dot(term_y)
        
    return F_wave

def vel_correction(K_global, M_global, Rx_global, Ry_global, coords, num_nodes, t_end, deltat, nu):
    # Manufactured solution
    u_exact, p_exact, F_exact_all_steps = exact_solution(coords, t_end, deltat, nu)
    print("Manufactured solution computed.")
    print("-" * 66)
    
    num_nodes_edge = int(np.sqrt(num_nodes))
    num_steps = int(np.floor(t_end / deltat))
    
    u_wave = np.zeros((num_nodes, 2 * num_steps))
    F_phi = np.zeros((num_nodes, num_steps))
    F_wave = np.zeros((num_nodes, 2 * num_steps))
    
    phi = np.zeros((num_nodes, num_steps))
    p = np.zeros((num_nodes, num_steps))
    
    p[:, 0] = p_exact[:, 0]
    u_wave[:, 0:2] = u_exact[:, 0:2]
    
    print(f"Total steps: {num_steps}, delta_t: {deltat}")
    
    KM = (3.0 / (2 * deltat)) * M_global + nu * K_global
    

    one_vec = np.ones((num_nodes, 1))
    
    c1 = sp.vstack([K_global, one_vec.T])
    c2 = sp.vstack([one_vec, np.array([[0]])]) 
    K_aug = sp.hstack([c1, c2]).tocsr()
    
    for k in range(num_steps - 1):
        
        is_startup = (k == 0)
        print(f"Step {k+1}/{num_steps}, time = {(k+1)*deltat:.4f}")
        
        idx_kp1_start = 2 * (k + 1)
        idx_kp1_end = 2 * (k + 2)
        idx_k_start = 2 * k
        idx_k_end = 2 * (k + 1)
        
        f_kp1 = F_exact_all_steps[:, idx_kp1_start:idx_kp1_end]
        f_k = F_exact_all_steps[:, idx_k_start:idx_k_end]
        
        u_wave_k = u_wave[:, idx_k_start:idx_k_end]
        
        if k >= 2:
            u_wave_km2 = u_wave[:, 2*(k-2):2*(k-1)]
            u_wave_km1 = u_wave[:, 2*(k-1):2*k]
        elif k == 1:
            u_wave_km2 = np.zeros((num_nodes, 2))
            u_wave_km1 = u_wave[:, 0:2]
        elif k == 0:
            u_wave_km1 = np.zeros((num_nodes, 2))
            u_wave_km2 = np.zeros((num_nodes, 2))

        # --- KROK 1: PHI ---
        rhs_phi_val = rhs_phi(f_kp1, f_k, u_wave_k, u_wave_km1, u_wave_km2, Rx_global, Ry_global, deltat, num_nodes, is_startup)
        
        b_aug = np.concatenate([rhs_phi_val, np.array([0])])
        
        # Solve
        sol = spla.spsolve(K_aug, b_aug)
        phi[:, k+1] = sol[0:num_nodes]
        
        print(f"  Step {k+1}: solved for phi, norm(phi) = {np.linalg.norm(phi[:, k+1]):.4f}")
        
        # --- KROK 2: PRESSURE ---
        div_u_wave = Rx_global.dot(u_wave_k[:, 0]) + Ry_global.dot(u_wave_k[:, 1])
        p[:, k+1] = p[:, k] + phi[:, k+1] - nu * div_u_wave
        print(f"  Step {k+1}: ||div(u)|| = {np.linalg.norm(div_u_wave):.3e}")
        
        # --- KROK 3: VELOCITY ---
        # Calculate RHS
        rhs_wave_val = rhs_u_wave(f_kp1, u_wave_k, u_wave_km1, p[:, k+1], M_global, Rx_global, Ry_global, deltat, num_nodes, is_startup)
        
        KM_mod = KM.tolil() 
        F_wave_mod = rhs_wave_val.copy() # (N, 2)
        
        # Exact solution for BCs at next step (k+1)
        u_exact_next = u_exact[:, idx_kp1_start:idx_kp1_end]
        
        def apply_bc(node_indices):
            for idx in node_indices:
                KM_mod[idx, :] = 0
                KM_mod[idx, idx] = 1
                F_wave_mod[idx, :] = u_exact_next[idx, :]

        bottom_nodes = np.arange(num_nodes_edge)
        
        top_start = (num_nodes_edge - 1) * num_nodes_edge
        top_nodes = np.arange(top_start, num_nodes)
        
        left_nodes = np.arange(0, num_nodes, num_nodes_edge)
        
        right_nodes = np.arange(num_nodes_edge - 1, num_nodes, num_nodes_edge)
        
        # Apply BCs
        all_bc_nodes = np.unique(np.concatenate([bottom_nodes, top_nodes, left_nodes, right_nodes]))
        apply_bc(all_bc_nodes)
        
        # Solve
        KM_mod_csr = KM_mod.tocsr()
        
        sol_wave = spla.spsolve(KM_mod_csr, F_wave_mod)
        u_wave[:, idx_kp1_start:idx_kp1_end] = sol_wave
        
    print("-" * 66)
    print("Velocity correction method finished solving")
    print("-" * 66)
    return phi, p, u_wave


def doplots(t_end, deltat, num_nodes, u_exact, p_exact, u_wave, p_vc, coords):
    num_nodes_edge = int(np.sqrt(num_nodes))
    
    
    x_grid = coords[:, 0].reshape((num_nodes_edge, num_nodes_edge))
    y_grid = coords[:, 1].reshape((num_nodes_edge, num_nodes_edge))
    
    num_steps = int(np.floor(t_end / deltat))
    t_vec = np.linspace(deltat, t_end, num_steps)
    
    # Errors
    error_u = np.zeros(num_steps)
    error_p = np.zeros(num_steps)
    error_u_L2 = np.zeros(num_steps)
    error_p_L2 = np.zeros(num_steps)
    
    for k in range(num_steps):
        # Indices
        idx_u = 2 * k
        idx_v = 2 * k + 1
        
        u_num_k = u_wave[:, idx_u]
        v_num_k = u_wave[:, idx_v]
        p_num_k = p_vc[:, k]
        
        u_ana_k = u_exact[:, idx_u]
        v_ana_k = u_exact[:, idx_v]
        p_ana_k = p_exact[:, k]
        
        u_diff_sq = (u_num_k - u_ana_k)**2 + (v_num_k - v_ana_k)**2
        u_ana_sq = u_ana_k**2 + v_ana_k**2
        
        if np.sum(u_ana_sq) > 1e-12:
            error_u[k] = np.max(np.sqrt(u_diff_sq))
            error_u_L2[k] = np.sqrt(np.sum(u_diff_sq)) / np.sqrt(np.sum(u_ana_sq))
            
        p_diff_sq = (p_num_k - p_ana_k)**2
        p_ana_sq = p_ana_k**2
        
        if np.sum(p_ana_sq) > 1e-12:
            error_p[k] = np.max(np.sqrt(p_diff_sq))

            error_p_L2[k] = np.sqrt(np.sum(p_diff_sq)) / np.sqrt(np.sum(p_ana_sq))
    
    # 1. Errors
    fig_err, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig_err.suptitle('Analiza Błędów')
    
    ax1.plot(t_vec, error_u, 'b-', linewidth=1.5)
    ax1.set_title('Błąd względny L2 dla prędkości')
    ax1.set_xlabel('Czas [s]')
    ax1.set_ylabel('błąd względny')
    ax1.grid(True)
    ax1.set_yscale('log')
    
    ax2.plot(t_vec, error_p, 'r-', linewidth=1.5)
    ax2.set_title('Błąd względny L2 dla ciśnienia')
    ax2.set_xlabel('Czas [s]')
    ax2.set_ylabel('błąd względny')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    # Final state data
    k_final = num_steps - 1
    
    u_num_final = u_wave[:, 2*k_final].reshape(num_nodes_edge, num_nodes_edge)
    v_num_final = u_wave[:, 2*k_final+1].reshape(num_nodes_edge, num_nodes_edge)
    mod_v_num = np.sqrt(u_num_final**2 + v_num_final**2)
    p_num_final = p_vc[:, k_final].reshape(num_nodes_edge, num_nodes_edge)
    
    u_ana_final = u_exact[:, 2*k_final].reshape(num_nodes_edge, num_nodes_edge)
    v_ana_final = u_exact[:, 2*k_final+1].reshape(num_nodes_edge, num_nodes_edge)
    mod_v_ana = np.sqrt(u_ana_final**2 + v_ana_final**2)
    p_ana_final = p_exact[:, k_final].reshape(num_nodes_edge, num_nodes_edge)
    
    # 2. Velocity Comparison
    fig_v = plt.figure(figsize=(14, 6))
    fig_v.suptitle(f'Porównanie modułu prędkości |v| dla t = {t_end:.2f} s')
    
    ax_v1 = fig_v.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax_v1.plot_surface(x_grid, y_grid, mod_v_num, cmap='viridis', edgecolor='none')
    ax_v1.set_title('Rozwiązanie Numeryczne')
    ax_v1.set_xlabel('x')
    ax_v1.set_ylabel('y')
    ax_v1.set_zlabel('|v|')
    fig_v.colorbar(surf1, ax=ax_v1, shrink=0.5)
    
    ax_v2 = fig_v.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax_v2.plot_surface(x_grid, y_grid, mod_v_ana, cmap='viridis', edgecolor='none')
    ax_v2.set_title('Rozwiązanie Analityczne')
    ax_v2.set_xlabel('x')
    ax_v2.set_ylabel('y')
    ax_v2.set_zlabel('|v|')
    fig_v.colorbar(surf2, ax=ax_v2, shrink=0.5)
    
    # 3. Pressure Comparison
    fig_p = plt.figure(figsize=(14, 6))
    fig_p.suptitle(f'Porównanie ciśnienia p dla t = {t_end:.2f} s')
    
    ax_p1 = fig_p.add_subplot(1, 2, 1, projection='3d')
    surf3 = ax_p1.plot_surface(x_grid, y_grid, p_num_final, cmap='viridis', edgecolor='none')
    ax_p1.set_title('Rozwiązanie Numeryczne')
    ax_p1.set_xlabel('x')
    ax_p1.set_ylabel('y')
    ax_p1.set_zlabel('p')
    fig_p.colorbar(surf3, ax=ax_p1, shrink=0.5)
    
    ax_p2 = fig_p.add_subplot(1, 2, 2, projection='3d')
    surf4 = ax_p2.plot_surface(x_grid, y_grid, p_ana_final, cmap='viridis', edgecolor='none')
    ax_p2.set_title('Rozwiązanie Analityczne')
    ax_p2.set_xlabel('x')
    ax_p2.set_ylabel('y')
    ax_p2.set_zlabel('p')
    fig_p.colorbar(surf4, ax=ax_p2, shrink=0.5)
    
    fig_emap, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(14, 6))
    fig_emap.suptitle(f'Błąd bezwzględny rozwiązania numerycznego dla t = {t_end:.2f} s')
    
    err_p_abs = np.abs(p_num_final - p_ana_final)
    c1 = ax_e1.pcolormesh(x_grid, y_grid, err_p_abs, cmap='parula' if 'parula' in plt.colormaps() else 'viridis', shading='auto')
    ax_e1.set_title("Błąd bezwzględny ciśnienia")
    ax_e1.set_xlabel('x')
    ax_e1.set_ylabel('y')
    ax_e1.axis('equal')
    fig_emap.colorbar(c1, ax=ax_e1, label=r'$\delta_p$')
    
    err_v_abs = np.abs(np.sqrt(u_num_final**2 + v_num_final**2) - np.sqrt(u_ana_final**2 + v_ana_final**2))
    c2 = ax_e2.pcolormesh(x_grid, y_grid, err_v_abs, cmap='parula' if 'parula' in plt.colormaps() else 'viridis', shading='auto')
    ax_e2.set_title("Błąd bezwzględny prędkości")
    ax_e2.set_xlabel('x')
    ax_e2.set_ylabel('y')
    ax_e2.axis('equal')
    fig_emap.colorbar(c2, ax=ax_e2, label=r'$\delta_u$')
    
    plt.show()
    
    return error_u_L2, error_p_L2, fig_p, fig_v, fig_err, fig_emap

if __name__ == "__main__":
    print("\n" * 2)
    start_time = time.time()
    
    print("-" * 66)
    try:
        x_in = input("Enter left and right x limit of fluid domain (e.g., 0 1): ")
        if not x_in.strip(): x_lim = [0, 1]
        else: x_lim = [float(x) for x in x_in.split()]
    except: x_lim = [0, 1]
    print(f"Entered x limits: {x_lim}")
    
    print("-" * 66)
    try:
        y_in = input("Enter lower and upper y limit of fluid domain (e.g., 0 1): ")
        if not y_in.strip(): y_lim = [0, 1]
        else: y_lim = [float(x) for x in y_in.split()]
    except: y_lim = [0, 1]
    print(f"Entered y limits: {y_lim}")
    
    print("-" * 66)
    try:
        n_in = input("Enter number of elements along one edge (e.g., 20): ")
        if not n_in.strip(): num_elm_edge = 20
        else: num_elm_edge = int(n_in)
    except: num_elm_edge = 20
    print(f"Specified number of elements along one edge: {num_elm_edge}")
    
    try:
        mp_in = input("Show mesh plot? Yes=1, No=0 (default 0): ")
        if not mp_in.strip(): meshplot = 0
        else: meshplot = int(mp_in)
    except: meshplot = 0
        
    print("-" * 66)
    coords, num_el, num_nodes, mesh_matrix, mesh_XY = make_mesh(x_lim, y_lim, num_elm_edge, meshplot)
    print(f'Built mesh consists of: {num_el} elements and {num_nodes} nodes')
    
    # Building matrices
    K_global = stiffness_matrix(num_nodes, num_el, mesh_matrix)
    M_global = mass_matrix(num_nodes, num_el, mesh_matrix)
    Rx_global, Ry_global = div_matrix(num_nodes, num_el, mesh_matrix)
    
    print("FEM matrices built")
    print("-" * 66)
    
    try:
        t_in = input("Define time duration of simulation (default 2.0): ")
        if not t_in.strip(): t_end = 2.0
        else: t_end = float(t_in)
    except: t_end = 2.0
        
    try:
        dt_in = input("Define time step for the simulation (default: 0.1): ")
        if not dt_in.strip(): deltat = 0.1
        else: deltat = float(dt_in)
    except: deltat = 0.1
    
    try:
        nu_in = input("Define kinetic viscosity of fluid (e.g., 0.01): ")
        if not nu_in.strip(): nu = 0.01
        else: nu = float(nu_in)
    except: nu = 0.01
        
    sim_name = input("Enter simulation name (default: test): ")
    if not sim_name.strip(): simulation_name = "test"
    else: simulation_name = sim_name
    
    # Run Solver
    phi_vc, p_vc, u_wave = vel_correction(K_global, M_global, Rx_global, Ry_global, coords, num_nodes, t_end, deltat, nu)
    
    u_exact, p_exact, F_exact = exact_solution(coords, t_end, deltat, nu)
    
    print("Generating plots")
    print("-" * 66)
    error_u_L2, error_p_L2, fig_p, fig_v, fig_err, fig_emap = doplots(t_end, deltat, num_nodes, u_exact, p_exact, u_wave, p_vc, coords)
    
    print("Saving mesh and results")
    print("-" * 66)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder_name = f"{timestamp}-{simulation_name}"
    output_path = os.path.join('.', 'Output_files', output_folder_name)
    os.makedirs(output_path, exist_ok=True)
    
    np.savez(os.path.join(output_path, 'results.npz'), 
             mesh_matrix=mesh_matrix, 
             coords=coords,
             p_exact=p_exact, u_exact=u_exact, F_exact=F_exact,
             p_vc=p_vc, u_wave=u_wave, phi_vc=phi_vc,
             error_u_L2=error_u_L2, error_p_L2=error_p_L2)
    
    # Save figures
    fig_err.savefig(os.path.join(output_path, 'error_plot.png'))
    fig_v.savefig(os.path.join(output_path, 'v_plot.png'))
    fig_p.savefig(os.path.join(output_path, 'p_plot.png'))
    fig_emap.savefig(os.path.join(output_path, 'error_map.png'))
    
    print("Program termination.")
    print("-" * 66)
    print(f"Total execution time: {time.time() - start_time:.2f} s")