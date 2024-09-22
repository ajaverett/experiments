# Chiastic Likelihood Calculator

Welcome to the **Chiastic Likelihood Calculator**! This Streamlit app calculates the reordering likelihood (**L**) and chiastic likelihood (**P**) for chiastic structures. It is designed to help researchers, literary analysts, and enthusiasts explore the statistical significance of chiasms in literary texts.

## Table of Contents

- [Introduction to Chiasmus](#introduction-to-chiasmus)
- [Mathematical Background](#mathematical-background)
  - [Reordering Likelihood (L)](#reordering-likelihood-l)
  - [Chiastic Likelihood (P)](#chiastic-likelihood-p)
- [Using the App](#using-the-app)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
  - [Input Parameters](#input-parameters)
  - [Calculating L and P](#calculating-l-and-p)
- [Examples](#examples)
  - [Example 1: Simple Chiasm](#example-1-simple-chiasm)
  - [Example 2: Complex Chiasm](#example-2-complex-chiasm)
- [Understanding the Results](#understanding-the-results)
- [Contributing](#contributing)
- [License](#license)

## Introduction to Chiasmus

**Chiasmus** is a rhetorical or literary figure in which words, grammatical constructions, or concepts are repeated in reverse order, in the same or modified form. The term comes from the Greek word "χιάζω" (chiázō), meaning "to shape like the letter Χ". Chiasmus often involves a sequence of ideas presented and then repeated in reverse order to form a symmetrical structure (ABBA pattern).

Chiasms are found in various forms of literature, including poetry, prose, and religious texts. Identifying chiasms can provide insights into the emphasis, structure, and meaning intended by the author.

However, determining whether a chiasm is intentional or a product of random arrangement requires statistical analysis. This app calculates the likelihood that a particular chiastic structure could have appeared by chance, helping users assess the significance of identified chiasms.

## Mathematical Background

### Reordering Likelihood ($L$)

The **reordering likelihood ($L$)** is the probability that a specific chiastic structure would appear by chance in a passage. It quantifies how likely it is for elements to arrange themselves in a chiastic pattern purely due to random ordering.

#### Simple Chiasms

For **simple chiasms**, where each element appears exactly twice and there are no extra repetitions or non-chiastic elements, $L$ can be calculated exactly using the formula:

$$
L = \frac{1}{(2n - 1)!!}
$$

- $n$: Number of chiastic elements
- $(2n - 1)!!$: Double factorial of $(2n - 1)$

#### Complex Chiasms

For **complex chiasms**, which may include extra repetitions of elements or non-chiastic elements, calculating $L$ analytically becomes challenging. Instead, we use **Monte Carlo simulations** to estimate $L$:

- Randomly rearrange the elements multiple times.
- Count the fraction of rearrangements that result in a chiastic structure.
- The estimated $L$ is the ratio of chiastic rearrangements to total rearrangements.

### Chiastic Likelihood (P)

The **chiastic likelihood (P)** represents the probability that, in a given number of opportunities, a certain number of chiastic structures would appear by chance. It is calculated using the binomial distribution, considering the reordering likelihood L as the success probability in each trial.

$$
P = 1 - \sum_{k=0}^{M-1} \binom{N}{k} L^k (1 - L)^{N - k}
$$

- $N$: Number of chiastic opportunities
- $M$: Number of chiastic instances observed
- $L$: Reordering likelihood

## Using the App

### Installation

1. **Prerequisites**:
   - Python 3.6 or higher installed on your system.
   - Streamlit library installed. If not, you can install it using:

```
pip install streamlit
```

2. **Clone or Download** the repository containing the app code.

3. **Navigate** to the directory containing the app script.

### Running the App

Run the app using the Streamlit CLI:


Replace `app.py` with the name of your script file if different.

### Input Parameters

#### Sidebar Inputs

- **Chiasm Name**: (Optional) A reference name for your chiasm analysis.
- **Number n of chiastic elements**: The number of unique elements participating in the chiasm.
- **Number of appearances of each chiastic element**: Enter the number of times each chiastic element appears, separated by commas.
- **Number m of nonchiastic elements**: The number of unique elements not participating in the chiasm.
- **Number of appearances of each nonchiastic element**: Enter the number of times each nonchiastic element appears, separated by commas.
- **Number r of rearrangements**: (For complex chiasms) The number of random permutations to perform in the Monte Carlo simulation. A higher number increases accuracy but takes longer to compute.
- **Number of duplicate levels**: (Optional) For complex chiasms where elements can be used more than once at different levels.

### Calculating L and P

1. **Input the parameters** in the sidebar according to your chiasm.
2. **Check if your chiasm is simple**:
   - If it is a simple chiasm (each element appears exactly twice, no extra elements), you can choose to calculate L exactly.
   - If not, proceed to calculate L using the Monte Carlo simulation.
3. **Click "Calculate L"** to compute the reordering likelihood.
4. **Optionally Calculate P**:
   - If you have multiple opportunities and observed instances of chiastic structures, you can calculate P.
   - Input the number of opportunities (N) and observed chiastic instances (M).
   - Click "Calculate P" to compute the chiastic likelihood.

## Examples

### Example 1: Simple Chiasm

Consider a simple chiasm with two elements, each appearing twice.

- **Chiastic Elements**: 2
- **Appearances**: 2, 2
- **Nonchiastic Elements**: 0

**Steps**:

1. Enter **2** for "Number n of chiastic elements".
2. Enter **2,2** for "Number of appearances of each chiastic element".
3. Enter **0** for "Number m of nonchiastic elements".
4. Since it's a simple chiasm, select **"Yes"** to calculate L exactly.
5. The app will display:
   - **Reordering likelihood L = 0.3333333333333333**
   - **Margin of error (+ or -) = 0.0000000000000000**

### Example 2: Complex Chiasm

Analyze a complex chiasm with:

- **Chiastic Elements**: 5
- **Appearances**: 2, 2, 2, 2, 2
- **Nonchiastic Elements**: 0
- **Number of Rearrangements**: 1,000,000

**Steps**:

1. Enter **5** for "Number n of chiastic elements".
2. Enter **2,2,2,2,2** for "Number of appearances of each chiastic element".
3. Enter **0** for "Number m of nonchiastic elements".
4. Since it's not a simple chiasm, proceed to Monte Carlo simulation.
5. Enter **1000000** for "Number r of rearrangements".
6. Click **"Calculate L"**.
7. The app will perform the simulation and display results similar to:
   - **Reordering likelihood L = 0.0010582010582011**
   - **Margin of error (+ or -) = (calculated value)**

## Understanding the Results

- **Reordering Likelihood (L)**:
  - A low L value indicates that the chiastic structure is unlikely to occur by random chance, suggesting intentional design.
  - A higher L value suggests the structure could easily occur randomly.
- **Chiastic Likelihood (P)**:
  - Represents the probability that the observed number of chiasms (M) would occur in the given number of opportunities (N) due to chance.
  - A low P value strengthens the argument that the chiasms are not coincidental.

**Margin of Error**:

- Provides an estimate of the uncertainty in the calculated L and P values.
- Smaller margins indicate more precise estimates.

## Contributing

Contributions to improve the app are welcome!

1. **Fork** the repository.
2. **Create** a new branch for your feature or bugfix.
3. **Commit** your changes with clear messages.
4. **Push** to your fork.
5. **Submit** a pull request describing your changes.

Please ensure that your contributions align with the project's goals and maintain code quality.

## Mathematical Derivations

### Double Factorial

The double factorial of an odd integer $n$ is defined as:

$$
n!! = n \times (n - 2) \times (n - 4) \times \dots \times 1
$$

For example:

$$
5!! = 5 \times 3 \times 1 = 15
$$

In the context of simple chiasms, the reordering likelihood is calculated using the double factorial to account for the decreasing number of ways to pair elements symmetrically.

### Monte Carlo Simulation

For complex chiasms, exact calculation of L is infeasible due to the combinatorial complexity. The Monte Carlo method estimates L by:

1. **Random Sampling**: Generate a large number of random permutations of the elements.
2. **Counting Successes**: Count how many of these permutations form a chiastic structure of the specified order.
3. **Estimating Probability**: Divide the number of successful permutations by the total number of permutations to estimate L.

The accuracy of the estimation improves with the number of permutations (rearrangements) used.

### Binomial Distribution for P

The chiastic likelihood P uses the cumulative binomial distribution:

\[
P = 1 - \sum_{k=0}^{M - 1} \binom{N}{k} L^k (1 - L)^{N - k}
\]

This formula calculates the probability of observing at least M chiastic structures in N opportunities, given the probability L of a chiastic structure occurring in a single opportunity.

- **\( \binom{N}{k} \)**: Binomial coefficient, representing the number of ways to choose k successes in N trials.
- **\( L^k \)**: Probability of k successes.
- **\( (1 - L)^{N - k} \)**: Probability of \( N - k \) failures.

## Practical Considerations

- **Computational Resources**: Large numbers of rearrangements (e.g., millions) may require significant computational time and resources.
- **Randomness**: The quality of the Monte Carlo simulation depends on the randomness of the permutations. The app uses Python's `random` module, which is suitable for most applications.
- **Interpreting Low P Values**: A very low P value suggests that the observed chiastic structures are unlikely to be due to chance alone. However, statistical significance should be considered alongside literary and contextual analysis.

## Contact and Support

If you encounter any issues or have questions:

- **Submit an Issue**: Use the GitHub repository's issue tracker to report bugs or request features.
- **Email**: [Your Email Address] for direct inquiries.

---

Thank you for using the Chiastic Likelihood Calculator! Your feedback and contributions are valuable in improving this tool for the literary and academic community.
