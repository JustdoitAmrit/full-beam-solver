import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

mpl_use("Agg")  # Use non-interactive backend for Streamlit compatibility

# Initialize or retrieve session state
if 'supports' not in st.session_state:
    st.session_state.supports = []  # Store supports persistently
if 'point_loads' not in st.session_state:
    st.session_state.point_loads = []  # Store point loads persistently
if 'udls' not in st.session_state:
    st.session_state.udls = []  # Store UDLs persistently

class Beam:
    def __init__(self, young, inertia, length, segments):
        self.young = young
        self.inertia = inertia
        self.length = length
        self.segments = segments
        self.node = self.generate_nodes()
        self.bar = self.generate_bars()
        self.point_load = np.zeros((self.segments + 1, 2))
        self.udl = np.zeros((self.segments + 1, 2))  # Store UDL effect
        self.force = np.zeros((self.segments, 4))
        self.displacement = np.zeros((self.segments, 4))

    def generate_nodes(self):
        node_positions = np.linspace(0, self.length, self.segments + 1)
        return np.array([[x, 0] for x in node_positions])

    def generate_bars(self):
        return np.array([[i, i + 1] for i in range(self.segments)])

    def apply_supports(self):
        support_matrix = np.ones((self.segments + 1, 2)).astype(int)
        for support in st.session_state.supports:
            position = support["position"]
            if support["type"] == "Fixed":
                support_matrix[position, :] = 0
            elif support["type"] == "Pinned":
                support_matrix[position, 0] = 0
        return support_matrix

    def apply_udl(self):
        # Reset UDL effects
        self.udl.fill(0)
        for udl in st.session_state.udls:
            start = udl["start_position"]
            end = udl["end_position"]
            magnitude = udl["magnitude"]
            self.udl[start:end + 1, 0] += magnitude  # Apply UDL to all affected nodes

    def analysis(self):
        self.apply_udl()  # Apply UDL effects before analysis
        self.support = self.apply_supports()  # Use the applied supports
        nn = len(self.node)
        ne = len(self.bar)
        n_dof = 2 * nn
        d = self.node[self.bar[:, 1], :] - self.node[self.bar[:, 0], :]
        length = np.sqrt((d**2).sum(axis=1))
        ss = np.zeros((n_dof, n_dof))
        
        for i in range(ne):
            l = length[i]
            k = self.young * self.inertia / l**3 * np.array(
                [[12, 6*l, -12, 6*l],
                 [6*l, 4*l**2, -6*l, 2*l**2],
                 [-12, -6*l, 12, -6*l],
                 [6*l, 2*l**2, -6*l, 4*l**2]]
            )
            aux = 2 * self.bar[i, :]
            index = np.r_[aux[0]:aux[0] + 2, aux[1]:aux[1] + 2]
            ss[np.ix_(index, index)] += k

        free_dof = self.support.flatten().nonzero()[0]
        kff = ss[np.ix_(free_dof, free_dof)]
        
        p = self.point_load.flatten() + self.udl.flatten()  # Include UDL effects
        pf = p[free_dof]
        uf = np.linalg.solve(kff, pf)
        
        u = self.support.astype(float).flatten()
        u[free_dof] = uf
        u = u.reshape(nn, 2)
        u_ele = np.concatenate((u[self.bar[:, 0]], u[self.bar[:, 1]]), axis=1)
        
        for i in range(ne):
            self.force[i] = np.dot(k, u_ele[i])
            self.displacement[i] = u_ele[i]

    def plot(self):
        fig, axs = plt.subplots(3, figsize=(10, 8))
        ne = len(self.bar)

        # Run analysis with current loads and UDLs
        self.analysis()

        for i in range(ne):
            xi, xf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
            yi, yf = self.node[self.bar[i, 0], 1], self.node[self.bar[i, 1], 1]
            axs[0].plot([xi, xf], [yi, yf], 'b', linewidth=1)

            dyi = yi + self.displacement[i, 0] * 100
            dyf = yf + self.displacement[i, 2] * 100
            axs[0].plot([xi, xf], [dyi, dyf], 'r', linewidth=2)

            mr_yi, mr_yf = -self.force[i, 1], self.force[i, 3]
            axs[1].plot([xi, xi, xf, xf], [0, mr_yi, mr_yf, 0], 'r', linewidth=1)
            axs[1].fill([xi, xi, xf, xf], [0, mr_yi, mr_yf, 0], 'c', alpha=0.3)

            fr_yi, fr_yf = -self.force[i, 0], self.force[i, 2]
            axs[2].plot([xi, xi, xf, xf], [0, fr_yi, fr_yf, 0], 'r', linewidth=1)
            axs[2].fill([xi, xi, xf, xf], [0, fr_yi, fr_yf, 0], 'c', alpha=0.3)

        axs[0].set_title("Beam Deflection")
        axs[0].set_xlabel("Length (m)")
        axs[0].set_ylabel("Deflection (m)")
        axs[0].grid()

        axs[1].set_title("Bending Moment Diagram (BMD)")
        axs[1].set_xlabel("Length (m)")
        axs[1].set_ylabel("Bending Moment (Nm)")
        axs[1].grid()

        axs[2].set_title("Shear Force Diagram (SFD)")
        axs[2].set_xlabel("Length (m)")
        axs[2].set_ylabel("Shear Force (N)")
        axs[2].grid()

        return fig

# Streamlit App Interface
st.title("Beam Analysis App")

st.sidebar.header("Beam Properties")
E = st.sidebar.number_input("Young's Modulus (E)", min_value=float(1e7), value=float(2e11), format="%.2e")
I = st.sidebar.number_input("Moment of Inertia (I)", min_value=float(1e-10), value=float(5e-4), format="%.2e")
length = st.sidebar.number_input("Beam Length (m)", min_value=float(1.0), value=float(10.0))
segments = st.sidebar.number_input("Number of Segments", min_value=int(1), value=int(12))

st.sidebar.header("Support Conditions")
support_type = st.sidebar.selectbox("Select Support Type", ["Fixed", "Pinned"])
support_position = st.sidebar.slider("Support Position", 0, segments, 0)
add_support = st.sidebar.button("Add Support")

if add_support:
    # Add support to session state for persistence
    st.session_state.supports.append({"position": support_position, "type": support_type})
    st.sidebar.write(f"{support_type} support added at position {support_position}")

st.sidebar.header("Point Loads")
load_magnitude = st.sidebar.number_input("Load Magnitude (N)", min_value=float(-1e6), value=float(-1000))
load_position = st.sidebar.slider("Load Position", 0, segments, 0)
add_load = st.sidebar.button("Add Load")

if add_load:
    # Add point load to session state for persistence
    st.session_state.point_loads.append({"position": load_position, "magnitude": load_magnitude})
    st.sidebar.write(f"Point load of {load_magnitude} N added at position {load_position}")

st.sidebar.header("Uniformly Distributed Load (UDL)")
udl_magnitude = st.sidebar.number_input("UDL Magnitude (N/m)", min_value=float(-1e6), value=float(100.0), format="%.2f")
udl_start_position = st.sidebar.slider("UDL Start Position", 0, segments, 0)
udl_end_position = st.sidebar.slider("UDL End Position", 0, segments, segments)

# Store UDL in session state
if st.sidebar.button("Add UDL"):
    st.session_state.udls.append({
        "magnitude": udl_magnitude,
        "start_position": udl_start_position,
        "end_position": udl_end_position,
    })
    st.sidebar.write(f"UDL of {udl_magnitude} N/m from {udl_start_position} to {udl_end_position}")

# Button to calculate and plot results
calculate_button = st.sidebar.button("Calculate")

if calculate_button:
    # Initialize Beam object with current parameters
    beam = Beam(E, I, length, segments)

    # Apply point loads
    for load in st.session_state.point_loads:
        beam.point_load[load["position"], 0] += load["magnitude"]

    # Generate plots
    fig = beam.plot()
    
    # Display the plots in Streamlit
    st.pyplot(fig)
