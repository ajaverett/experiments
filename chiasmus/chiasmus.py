import streamlit as st
import random
import math

# Function to generate a random permutation and compute instances after position
def permute(elements_list, appearances_per_element, total_num_elements, total_elements):
    ll = elements_list.copy()
    permuted_elements_list = []
    instances_after_position = [[0] * (total_elements + 1) for _ in range(total_num_elements + 1)]

    # Initialize instances_after_position[0] with counts appearances_per_element
    for j in range(1, total_elements + 1):
        instances_after_position[0][j] = appearances_per_element[j - 1]

    # Generate a random permutation
    for i in range(total_num_elements, 0, -1):
        j = random.randint(0, i - 1)
        permuted_elements_list.append(ll[j])
        del ll[j]

    permuted_elements_list = permuted_elements_list[::-1]  # Reverse to get correct order

    # Compute instances_after_position
    for i in range(1, total_num_elements + 1):
        for j in range(1, total_elements + 1):
            instances_after_position[i][j] = instances_after_position[i - 1][j]
        instances_after_position[i][permuted_elements_list[i - 1]] -= 1

    return permuted_elements_list, instances_after_position

# Function to calculate chiastic likelihood P
def find_p(num_chiastic_opportunities, num_chiastic_instances, L):
    pii = (1.0 - L) ** num_chiastic_opportunities
    P = 1.0 - pii
    if num_chiastic_instances > 1:
        ratio = L / (1.0 - L)
        for i in range(1, num_chiastic_instances):
            pii *= (num_chiastic_opportunities + 1 - i) * ratio / i
            P -= pii
    return P

# Main Streamlit app
def main():
    st.title("Chiastic Likelihood Calculator")
    st.write("This app calculates the reordering likelihood **L** for chiastic structures.")
    st.write("It also calculates the chiastic likelihood **P** if desired.")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")

    # Chiasm name
    chiasm_name = st.sidebar.text_area("""Chiasm Name""")

    # Number of chiastic elements
    num_chiastic_elements = st.sidebar.number_input(
        "Number n of chiastic elements", min_value=1, max_value=100, step=1, value=2
    )

    # Appearances of each chiastic element
    chiastic_appearances_input = st.sidebar.text_input(
        "Number of appearances of each chiastic element (comma-separated)", value="2,2"
    )

    # Number of nonchiastic elements
    num_nonchiastic_elements = st.sidebar.number_input(
        "Number m of nonchiastic elements", min_value=0, max_value=100, step=1, value=0
    )

    # Appearances of each nonchiastic element
    if num_nonchiastic_elements > 0:
        nonchiastic_appearances_input = st.sidebar.text_input(
            "Number of appearances of each nonchiastic element (comma-separated)", value=""
        )
    else:
        nonchiastic_appearances_input = ""

    # Initialize variables
    is_simple_chiasm = False
    total_num_elements = 0
    elements_list = []
    appearances_per_element = []

    # Parse chiastic appearances
    try:
        appearances_per_element = [
            int(k) for k in chiastic_appearances_input.replace(',', ' ').split()
        ]
        if len(appearances_per_element) != num_chiastic_elements:
            st.error(f"Please enter exactly {num_chiastic_elements} numbers for chiastic elements.")
            return
        if any(k < 2 for k in appearances_per_element):
            st.error("Each chiastic element must appear at least twice.")
            return
    except ValueError:
        st.error("Invalid input for chiastic appearances. Please enter integers separated by commas.")
        return

    # Parse nonchiastic appearances
    if num_nonchiastic_elements > 0:
        try:
            nonchiastic_appearances = [
                int(k) for k in nonchiastic_appearances_input.replace(',', ' ').split()
            ]
            if len(nonchiastic_appearances) != num_nonchiastic_elements:
                st.error(f"Please enter exactly {num_nonchiastic_elements} numbers for nonchiastic elements.")
                return
            if any(k < 2 for k in nonchiastic_appearances):
                st.error("Each nonchiastic element must appear at least twice.")
                return
            appearances_per_element.extend(nonchiastic_appearances)
        except ValueError:
            st.error("Invalid input for nonchiastic appearances. Please enter integers separated by commas.")
            return
    else:
        nonchiastic_appearances = []

    total_elements = num_chiastic_elements + num_nonchiastic_elements

    # Build the list of elements in standard order
    num_possible_levels = 0
    for j in range(len(appearances_per_element)):
        num_possible_levels += appearances_per_element[j] // 2
        for _ in range(appearances_per_element[j]):
            total_num_elements += 1
            if total_num_elements > 200:
                st.error("Total number of element appearances exceeds 200. Please try again.")
                return
            elements_list.append(j + 1)  # Elements are 1-indexed

    # Check if the chiasm is simple
    is_simple_chiasm = (
        num_nonchiastic_elements == 0 and total_num_elements == num_chiastic_elements * 2
    )

    if is_simple_chiasm:
        calc_exact = st.sidebar.radio(
            "This chiasm is simple. Calculate L exactly?", ("Yes", "No"), index=0
        )
        if calc_exact == "Yes":
            L = 1.0
            for i in range(1, num_chiastic_elements + 1):
                L /= (2 * i - 1)
            error_margin = 0.0
            st.write(f"Reordering likelihood **L** = {L:.16f}")
            st.write(f"Margin of error (+ or -) = {error_margin:.16f}")

            # Store L and error_margin in session state
            st.session_state['L'] = L
            st.session_state['error_margin'] = error_margin

        else:
            L = None  # Will be calculated later
    else:
        calc_exact = "No"
        L = None  # Will be calculated later

    # Additional inputs for complex chiasms
    if calc_exact == "No":
        num_rearrangements = st.sidebar.number_input(
            "Number r of rearrangements", min_value=1, step=1, value=10000
        )
        num_duplicate_levels = 0
        if num_possible_levels > num_chiastic_elements:
            num_duplicate_levels = st.sidebar.number_input(
                "Number of duplicate levels (normally 0)", min_value=0, step=1, value=0
            )
        else:
            num_duplicate_levels = 0

        # Calculate L using Monte Carlo simulation
        calculate_button = st.sidebar.button("Calculate L")
        if calculate_button:
            with st.spinner("Calculating..."):
                allow_duplicate_elements = False
                num_possible_chiastic_elements = total_elements
                if num_duplicate_levels > 0:
                    allow_duplicate_elements = True
                    num_possible_chiastic_elements = total_num_elements // 2
                    num_chiastic_elements += num_duplicate_levels

                permutations_per_order = [0] * (num_possible_chiastic_elements + 1)
                cumulative_permutations_per_order = [0] * (num_possible_chiastic_elements + 1)

                total_iterations = int(num_rearrangements)
                progress_bar = st.progress(0)
                for iteration in range(total_iterations):
                    # Update progress bar
                    if iteration % max(1, total_iterations // 100) == 0:
                        progress_bar.progress((iteration + 1) / total_iterations)

                    # Generate random permutation
                    permuted_elements_list, instances_after_position = permute(
                        elements_list, appearances_per_element, total_num_elements, total_elements
                    )

                    used_elements = [0] * (total_elements + 1)
                    indices_in_combination = [0] * (2 * num_possible_chiastic_elements + 2)
                    c = [0] * (num_possible_chiastic_elements + 1)
                    indices_in_combination[2 * num_possible_chiastic_elements + 1] = total_num_elements + 1
                    k = 1
                    indices_in_combination[k] = 1
                    finished = False
                    n = 0  # Highest chiastic order found so far
                    while not finished:
                        j = permuted_elements_list[indices_in_combination[k] - 1]
                        k2 = 2 * num_possible_chiastic_elements - k + 2
                        i = indices_in_combination[k2] - 1
                        if i <= indices_in_combination[k]:
                            nmax = -1
                        else:
                            nmax = k - 1
                            for jj in range(1, total_elements + 1):
                                njj = instances_after_position[indices_in_combination[k] - 1][jj] - \
                                      instances_after_position[i][jj]
                                if njj > 1 and used_elements[jj] == 0:
                                    nmax += 1
                            nj = instances_after_position[indices_in_combination[k] - 1][j] - \
                                 instances_after_position[i][j]
                        if nmax <= n:
                            if k == 1:
                                finished = True
                                permutations_per_order[n] += 1
                            else:
                                k -= 1
                                used_elements[permuted_elements_list[indices_in_combination[k] - 1]] = 0
                                indices_in_combination[k] += 1
                        elif used_elements[j] == 1 or nj < 2:
                            indices_in_combination[k] += 1
                        else:
                            while permuted_elements_list[i - 1] != j:
                                i -= 1
                            if k > n:
                                n = k
                                for kp in range(1, n + 1):
                                    c[kp] = permuted_elements_list[indices_in_combination[kp] - 1]
                            indices_in_combination[2 * num_possible_chiastic_elements - k + 1] = i
                            if not allow_duplicate_elements:
                                used_elements[j] = 1
                            if k == num_possible_chiastic_elements:
                                finished = True
                                permutations_per_order[n] += 1
                            else:
                                k += 1
                                indices_in_combination[k] = indices_in_combination[k - 1] + 1

                cumulative_permutations_per_order[num_possible_chiastic_elements] = permutations_per_order[num_possible_chiastic_elements]
                for n_idx in range(num_possible_chiastic_elements - 1, 0, -1):
                    cumulative_permutations_per_order[n_idx] = cumulative_permutations_per_order[n_idx + 1] + permutations_per_order[n_idx]

                L = cumulative_permutations_per_order[num_chiastic_elements] / num_rearrangements
                error_margin = math.sqrt(cumulative_permutations_per_order[num_chiastic_elements]) / num_rearrangements
                st.success("Calculation completed!")
                st.write(f"Reordering likelihood **L** = {L:.16f}")
                st.write(f"Margin of error (+ or -) = {error_margin:.16f}")

                # Store L and error_margin in session state
                st.session_state['L'] = L
                st.session_state['error_margin'] = error_margin

    # Option to calculate P
    if 'L' in st.session_state and st.session_state['L'] is not None:
        L = st.session_state['L']
        error_margin = st.session_state['error_margin']

        calc_P = st.sidebar.radio("Calculate P?", ("No", "Yes"), index=0)
        if calc_P == "Yes":
            num_chiastic_opportunities = st.sidebar.number_input(
                "Number N of chiastic opportunities", min_value=1, step=1, value=1
            )
            num_chiastic_instances = st.sidebar.number_input(
                "Number M of these that are chiastic", min_value=1, step=1, value=1
            )
            calculate_P_button = st.sidebar.button("Calculate P")
            if calculate_P_button:
                P = find_p(num_chiastic_opportunities, num_chiastic_instances, L)
                if L + error_margin < 1.0:
                    P_error = find_p(num_chiastic_opportunities, num_chiastic_instances, L + error_margin)
                elif L - error_margin > 0.0:
                    P_error = find_p(num_chiastic_opportunities, num_chiastic_instances, L - error_margin)
                else:
                    P_error = 100.0
                P_error_margin = abs(P - P_error)
                st.write(f"Chiastic likelihood **P** = {P:.16f}")
                st.write(f"Margin of error (+ or -) = {P_error_margin:.16f}")
    else:
        st.sidebar.write("Please calculate L first.")

if __name__ == '__main__':
    main()
