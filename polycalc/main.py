import streamlit as st

# Title and description
st.title("Mesh Weight Calculator")
st.write("""
This calculator computes the **weight per square meter** of an HDPE mesh based on the provided yarn counts and deniers.
""")

# Input parameters
st.header("Input Parameters")

warp_count = st.number_input("Warp Count (yarns per inch)", min_value=0.0, value=10.0)
warp_denier = st.number_input("Warp Denier", min_value=0.0, value=1000.0)
weft_count = st.number_input("Weft Count (yarns per inch)", min_value=0.0, value=10.0)
weft_denier = st.number_input("Weft Denier", min_value=0.0, value=1000.0)
density_hdpe = st.number_input("Density of HDPE (g/cm³)", min_value=0.0, value=0.935)

# Constants
INCH_TO_METER = 0.0254  # Conversion factor from inches to meters

# Calculations
st.header("Calculations")

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
total_mass_per_sqm = (warp_length_per_sqm * warp_mass_per_meter) + (weft_length_per_sqm * weft_mass_per_meter)

st.header("Result")
st.write(f"The **weight per square meter** of the mesh is:")
st.subheader(f"{total_mass_per_sqm:.2f} grams")

# LaTeX Calculations
st.header("Detailed Calculations")

st.write("### Step 1: Convert Yarns per Inch to Yarns per Meter")

st.latex(r"""
\text{Yarns per Meter} = \frac{\text{Yarns per Inch}}{0.0254}
""")

st.latex(rf"""
\text{{Warp Yarns per Meter}} = \frac{{{warp_count}}}{{0.0254}} = {warp_yarns_per_meter:.2f} \text{{ yarns/m}}
""")

st.latex(rf"""
\text{{Weft Yarns per Meter}} = \frac{{{weft_count}}}{{0.0254}} = {weft_yarns_per_meter:.2f} \text{{ yarns/m}}
""")

st.write("### Step 2: Calculate Mass per Meter of the Yarns")

st.latex(r"""
\text{Mass per Meter (g/m)} = \frac{\text{Denier}}{9000}
""")

st.latex(rf"""
\text{{Warp Mass per Meter}} = \frac{{{warp_denier}}}{{9000}} = {warp_mass_per_meter:.4f} \text{{ g/m}}
""")

st.latex(rf"""
\text{{Weft Mass per Meter}} = \frac{{{weft_denier}}}{{9000}} = {weft_mass_per_meter:.4f} \text{{ g/m}}
""")

st.write("### Step 3: Calculate the Total Length of Yarn per Square Meter")

st.latex(r"""
\text{Total Length (m)} = \text{Yarns per Meter} \times 1 \text{ meter}
""")

st.latex(rf"""
\text{{Warp Length per Square Meter}} = {warp_yarns_per_meter:.2f} \times 1 = {warp_length_per_sqm:.2f} \text{{ m}}
""")

st.latex(rf"""
\text{{Weft Length per Square Meter}} = {weft_yarns_per_meter:.2f} \times 1 = {weft_length_per_sqm:.2f} \text{{ m}}
""")

st.write("### Step 4: Calculate the Total Mass per Square Meter")

st.latex(r"""
\text{Total Mass (g)} = (\text{Warp Length} \times \text{Warp Mass per Meter}) + (\text{Weft Length} \times \text{Weft Mass per Meter})
""")

st.latex(rf"""
\text{{Total Mass}} = ({warp_length_per_sqm:.2f} \times {warp_mass_per_meter:.4f}) + ({weft_length_per_sqm:.2f} \times {weft_mass_per_meter:.4f}) = {total_mass_per_sqm:.2f} \text{{ grams}}
""")



st.write("""
""")


st.write("""
***
""")

st.write("""
***
""")


st.write("""
         Therefore, the **weight per square meter** of the mesh is:  
""")


st.header(f"{total_mass_per_sqm:.2f} grams")

st.write("""
***
""")