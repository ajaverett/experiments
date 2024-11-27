import streamlit as st

# Title and description
st.title("Material Weight Calculator")
st.write("""
This calculator computes the **weight per square meter** of a mesh or LDPE film based on the provided parameters.
""")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Mesh inputs
st.sidebar.subheader("Mesh Parameters")
warp_count = st.sidebar.number_input("Warp Count (yarns per inch)", min_value=0.0, value=10.0)
warp_denier = st.sidebar.number_input("Warp Denier", min_value=0.0, value=1000.0)
weft_count = st.sidebar.number_input("Weft Count (yarns per inch)", min_value=0.0, value=10.0)
weft_denier = st.sidebar.number_input("Weft Denier", min_value=0.0, value=1000.0)

# LDPE inputs
st.sidebar.subheader("LDPE Film Parameters")
ldpe_thickness = st.sidebar.number_input("LDPE Thickness (microns)", min_value=0.0, value=50.0)
ldpe_density = st.sidebar.number_input("LDPE Density (g/cm³)", min_value=0.0, value=0.92)

# Constants
INCH_TO_METER = 0.0254  # Conversion factor from inches to meters

# Mesh Calculations
st.header("Mesh Calculations")

# Step 1: Convert yarns per inch to yarns per meter
warp_yarns_per_meter = warp_count / INCH_TO_METER
weft_yarns_per_meter = weft_count / INCH_TO_METER

st.write(f"**Warp Yarns per Meter**: {warp_yarns_per_meter:.2f} yarns/m")
st.write(f"**Weft Yarns per Meter**: {weft_yarns_per_meter:.2f} yarns/m")

# Step 2: Calculate mass per meter of the yarns (grams per meter)
warp_mass_per_meter = warp_denier / 9000  # Denier is grams per 9000 meters
weft_mass_per_meter = weft_denier / 9000

st.write(f"**Warp Mass per Meter**: {warp_mass_per_meter:.4f} g/m")
st.write(f"**Weft Mass per Meter**: {weft_mass_per_meter:.4f} g/m")

# Step 3: Calculate the total length of yarn per square meter (warp and weft)
warp_length_per_sqm = warp_yarns_per_meter * 1  # Length in meters for 1 square meter
weft_length_per_sqm = weft_yarns_per_meter * 1

st.write(f"**Warp Length per Square Meter**: {warp_length_per_sqm:.2f} m")
st.write(f"**Weft Length per Square Meter**: {weft_length_per_sqm:.2f} m")

# Step 4: Calculate the total mass per square meter
total_mass_per_sqm_mesh = (warp_length_per_sqm * warp_mass_per_meter) + (weft_length_per_sqm * weft_mass_per_meter)

st.write(f"The **weight per square meter** of the mesh is:")
st.subheader(f"{total_mass_per_sqm_mesh:.2f} grams")

# LDPE Film Calculations
st.header("LDPE Film Calculations")

# Weight of LDPE film in grams per square meter
ldpe_weight_per_sqm = ldpe_thickness * ldpe_density * 0.1

st.write(f"**LDPE Thickness**: {ldpe_thickness} microns")
st.write(f"**LDPE Density**: {ldpe_density} g/cm³")
st.write(f"The **weight per square meter** of the LDPE film is:")
st.subheader(f"{ldpe_weight_per_sqm:.2f} grams")

st.write("""
***
""")

show_mesh_calcs = st.button("Show Calculations for Mesh Parameters")
show_ldpe_calcs = st.button("Show Calculations for LDPE Film Parameters")


if show_mesh_calcs:
    st.subheader("Mesh Calculations - Step-by-Step")
    st.latex(r"""
    \text{Warp Yarns per Meter} = \frac{\text{Warp Count (yarns per inch)}}{0.0254}
    """)
    st.latex(rf"""
    = \frac{{{warp_count}}}{{0.0254}} = {warp_yarns_per_meter:.2f} \, \text{{yarns/m}}
    """)
    st.latex(r"""
    \text{Weft Yarns per Meter} = \frac{\text{Weft Count (yarns per inch)}}{0.0254}
    """)
    st.latex(rf"""
    = \frac{{{weft_count}}}{{0.0254}} = {weft_yarns_per_meter:.2f} \, \text{{yarns/m}}
    """)
    st.latex(r"""
    \text{Warp Mass per Meter} = \frac{\text{Warp Denier}}{9000}
    """)
    st.latex(rf"""
    = \frac{{{warp_denier}}}{{9000}} = {warp_mass_per_meter:.4f} \, \text{{g/m}}
    """)
    st.latex(r"""
    \text{Weft Mass per Meter} = \frac{\text{Weft Denier}}{9000}
    """)
    st.latex(rf"""
    = \frac{{{weft_denier}}}{{9000}} = {weft_mass_per_meter:.4f} \, \text{{g/m}}
    """)
    st.latex(r"""
    \text{Total Mass per Square Meter} = 
    (\text{Warp Length per Square Meter} \times \text{Warp Mass per Meter}) + 
    (\text{Weft Length per Square Meter} \times \text{Weft Mass per Meter})
    """)
    st.latex(rf"""
    = ({warp_length_per_sqm:.2f} \times {warp_mass_per_meter:.4f}) + 
    ({weft_length_per_sqm:.2f} \times {weft_mass_per_meter:.4f}) = 
    {total_mass_per_sqm_mesh:.2f} \, \text{{grams}}
    """)

if show_ldpe_calcs:
    st.subheader("LDPE Film Calculations - Step-by-Step")
    st.latex(r"""
    \text{Weight (GSM)} = \text{Thickness (microns)} \times \text{Density (g/cm}^3\text{)} \times 0.1
    """)
    st.latex(rf"""
    = {ldpe_thickness} \times {ldpe_density} \times 0.1 = {ldpe_weight_per_sqm:.2f} \, \text{{grams}}
    """)





