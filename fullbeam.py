import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

mpl_use("Agg")  # Use non-interactive backend for Streamlit compatibility

# Initialize or retrieve supports and loads lists in session state
if 'supports' not in st.session_state:
    st.session_state.supports = []  # Store supports persistently

if 'point_loads' not in st.session_state:
    st.session_state.point_loads = []  # Store point loads persistently

if 'udls' not in st.session_state:
    st.session_state.udls = []  # Store uniformly distributed loads persistently

class Beam:
    def __init__(self, young, inertia, length, segments):
        self.young = young
        self.inertia = inertia
        self.length = length
        self.segments = segments
        self.node = self.generate_nodes()
        self.bar = self.generate_bars()
        self.point_load = np.zeros((self.segments + 1, 2))
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

    def apply_point_loads(self):
        self.point_load.fill(0)
        for load in st.session_state.point_loads:
            position = load["position"]
            magnitude = load["magnitude"]
            self.point_load[position, 0] += magnitude  # Accumulate point loads at the same position

    def apply_udls(self):
        for udl in st.session_state.udls:
            position = udl["position"]
            magnitude = udl["magnitude"]
            start_node = position
            end_node = position + 1
            if end_node <= self.segments:
                # Distribute the load evenly across the segment
                self.point_load[start_node, 0] += magnitude / 2  # Half at the start
                self.point_load[end_node, 0] += magnitude / 2  # Half at the end

    def analysis(self):
        self.apply_point_loads()  # Apply the point loads
        self.apply_udls()         # Apply the UDLs
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
        p = self.point_load.flatten()
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

        # Run analysis (this uses the applied loads from the session state)
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
E = st.sidebar.number_input("Young's Modulus (E)", min_value=1e7, value=2e11, format="%.2e")
I = st.sidebar.number_input("Moment of Inertia (I)", min_value=1e-10, value=5e-4, format="%.2e")
length = st.sidebar.number_input("Beam Length (m)", min_value=1.0, value=10.0)
segments = st.sidebar.number_input("Number of Segments", min_value=1, value=12)

st.sidebar.header("Support Conditions")
support_type = st.sidebar.selectbox("Select Support Type", ["Fixed", "Pinned"])
support_position = st.sidebar.slider("Support Position", 0, segments, 0)
add_support = st.sidebar.button("Add Support")

st.sidebar.header("Point Load Conditions")
load_position = st.sidebar.slider("Point Load Position", 0, segments, 0)
load_magnitude = st.sidebar.number_input("Point Load Magnitude (N)", min_value=-1e6, value=-1000.0)  # Changed to float
add_point_load = st.sidebar.button("Add Point Load")

st.sidebar.header("Uniformly Distributed Load Conditions")
udl_position = st.sidebar.slider("UDL Position", 0, segments - 1, 0)  # Position for UDL start
udl_magnitude = st.sidebar.number_input("UDL Magnitude (N/m)", min_value=0.0, value=500.0)  # Magnitude of UDL
add_udl = st.sidebar.button("Add UDL")

# Initialize beam instance
beam = Beam(E, I, length, segments)

if add_support:
    # Add support to session state for persistence
    st.session_state.supports.append({"position": support_position, "type": support_type})
    st.sidebar.write(f"{support_type} support added at position {support_position}")

if add_point_load:
    # Add point load to session state for persistence
    st.session_state.point_loads.append({"position": load_position, "magnitude": load_magnitude})
    st.sidebar.write(f"Point load of {load_magnitude} N added at position {load_position}")

if add_udl:
    # Add UDL to session state for persistence
    st.session_state.udls.append({"position": udl_position, "magnitude": udl_magnitude})
    st.sidebar.write(f"UDL of {udl_magnitude} N/m added starting at position {udl_position}")

# Display added supports and loads
st.sidebar.subheader("Current Supports")
for support in st.session_state.supports:
    st.sidebar.write(f"{support['type']} at position {support['position']}")

st.sidebar.subheader("Current Point Loads")
for load in st.session_state.point_loads:
    st.sidebar.write(f"{load['magnitude']} N at position {load['position']}")

st.sidebar.subheader("Current UDLs")
for udl in st.session_state.udls:
    st.sidebar.write(f"{udl['magnitude']} N/m starting at position {udl['position']}")

st.subheader("Beam Analysis Results")
fig = beam.plot()
st.pyplot(fig)