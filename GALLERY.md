# Reasoning Gym Dataset Gallery
This gallery shows examples from all available datasets using their default configurations.

## Available Datasets (101)
Legend: ✅ = Has curriculum, ❌ = No curriculum

- [ab](#ab) ✅
- [acre](#acre) ❌
- [advanced_geometry](#advanced_geometry) ✅
- [aiw](#aiw) ✅
- [arc_1d](#arc_1d) ❌
- [arc_agi](#arc_agi) ❌
- [base_conversion](#base_conversion) ✅
- [basic_arithmetic](#basic_arithmetic) ✅
- [bf](#bf) ✅
- [binary_alternation](#binary_alternation) ✅
- [binary_matrix](#binary_matrix) ✅
- [bitwise_arithmetic](#bitwise_arithmetic) ✅
- [boxnet](#boxnet) ✅
- [caesar_cipher](#caesar_cipher) ✅
- [calendar_arithmetic](#calendar_arithmetic) ✅
- [chain_sum](#chain_sum) ✅
- [circuit_logic](#circuit_logic) ✅
- [codeio](#codeio) ❌
- [color_cube_rotation](#color_cube_rotation) ✅
- [complex_arithmetic](#complex_arithmetic) ✅
- [count_bits](#count_bits) ✅
- [count_primes](#count_primes) ✅
- [countdown](#countdown) ❌
- [course_schedule](#course_schedule) ✅
- [cryptarithm](#cryptarithm) ✅
- [decimal_arithmetic](#decimal_arithmetic) ✅
- [decimal_chain_sum](#decimal_chain_sum) ✅
- [dice](#dice) ✅
- [emoji_mystery](#emoji_mystery) ✅
- [family_relationships](#family_relationships) ✅
- [figlet_font](#figlet_font) ❌
- [fraction_simplification](#fraction_simplification) ✅
- [futoshiki](#futoshiki) ❌
- [game_of_life](#game_of_life) ✅
- [game_of_life_halting](#game_of_life_halting) ✅
- [gcd](#gcd) ❌
- [graph_color](#graph_color) ✅
- [group_anagrams](#group_anagrams) ✅
- [gsm_symbolic](#gsm_symbolic) ❌
- [intermediate_integration](#intermediate_integration) ✅
- [isomorphic_strings](#isomorphic_strings) ✅
- [jugs](#jugs) ✅
- [knight_swap](#knight_swap) ❌
- [knights_knaves](#knights_knaves) ❌
- [largest_island](#largest_island) ✅
- [lcm](#lcm) ✅
- [leg_counting](#leg_counting) ✅
- [letter_counting](#letter_counting) ✅
- [letter_jumble](#letter_jumble) ✅
- [list_functions](#list_functions) ❌
- [mahjong_puzzle](#mahjong_puzzle) ✅
- [manipulate_matrix](#manipulate_matrix) ✅
- [maze](#maze) ✅
- [mini_sudoku](#mini_sudoku) ✅
- [modulo_grid](#modulo_grid) ✅
- [n_queens](#n_queens) ✅
- [needle_haystack](#needle_haystack) ✅
- [number_filtering](#number_filtering) ✅
- [number_format](#number_format) ✅
- [number_sequence](#number_sequence) ✅
- [number_sorting](#number_sorting) ✅
- [palindrome_generation](#palindrome_generation) ✅
- [palindrome_partitioning](#palindrome_partitioning) ✅
- [polynomial_equations](#polynomial_equations) ✅
- [polynomial_multiplication](#polynomial_multiplication) ✅
- [pool_matrix](#pool_matrix) ✅
- [power_function](#power_function) ✅
- [prime_factorization](#prime_factorization) ✅
- [products](#products) ✅
- [propositional_logic](#propositional_logic) ✅
- [puzzle24](#puzzle24) ❌
- [quantum_lock](#quantum_lock) ✅
- [ransom_note](#ransom_note) ✅
- [rearc](#rearc) ✅
- [rectangle_count](#rectangle_count) ✅
- [rotate_matrix](#rotate_matrix) ✅
- [rotten_oranges](#rotten_oranges) ✅
- [rubiks_cube](#rubiks_cube) ✅
- [rush_hour](#rush_hour) ✅
- [self_reference](#self_reference) ✅
- [sentence_reordering](#sentence_reordering) ✅
- [shortest_path](#shortest_path) ✅
- [simple_equations](#simple_equations) ✅
- [simple_geometry](#simple_geometry) ✅
- [simple_integration](#simple_integration) ✅
- [sokoban](#sokoban) ✅
- [spell_backward](#spell_backward) ✅
- [spiral_matrix](#spiral_matrix) ✅
- [string_insertion](#string_insertion) ✅
- [string_manipulation](#string_manipulation) ✅
- [string_splitting](#string_splitting) ✅
- [string_synthesis](#string_synthesis) ✅
- [sudoku](#sudoku) ✅
- [syllogism](#syllogism) ❌
- [time_intervals](#time_intervals) ✅
- [tower_of_hanoi](#tower_of_hanoi) ✅
- [tsumego](#tsumego) ✅
- [word_ladder](#word_ladder) ✅
- [word_sequence_reversal](#word_sequence_reversal) ✅
- [word_sorting](#word_sorting) ❌
- [zebra_puzzles](#zebra_puzzles) ✅

## Dataset Examples
### ab
Generates A::B tasks, as described by @VictorTaelin [here](https://x.com/VictorTaelin/status/1776096481704804789)

Default configuration:
```python
seed = 42
size = 500
length = 10
```

Example tasks:
````
Example 1:
Question: A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule.

Now, consider the following program:

A# A# #A B# B# B# A# A# #B A#

Return the final state of the program.

Answer: A# B# B# A# A# A#
Metadata: {'difficulty': {'length': 10}}

Example 2:
Question: A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule.

Now, consider the following program:

A# #A B# #B #A A# #B #B A# #B

Return the final state of the program.

Answer: #A #B #B #B A# A#
Metadata: {'difficulty': {'length': 10}}

Example 3:
Question: A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule.

Now, consider the following program:

#B A# B# #B B# #A A# B# A# A#

Return the final state of the program.

Answer: #B B# A# B# A# A#
Metadata: {'difficulty': {'length': 10}}

````

### acre
Default configuration:
```python
train = 1
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: You are a researcher studying causal relationships using Blicket experiments. In these experiments, certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with:
- "on" if you can determine the detector light will turn on
- "off" if you can determine the detector light will stay off
- "undetermined" if there is insufficient evidence to reach a conclusion

Do not use quotation marks in your answer.

Previous experimental results:
yellow rubber cylinder → on
red rubber sphere → off
yellow rubber cylinder, red rubber sphere → on
yellow metal sphere, red metal cylinder, brown rubber cylinder, purple rubber sphere, yellow rubber cube → on
yellow rubber cube, brown rubber cylinder, purple rubber sphere → off
yellow metal sphere, red metal cylinder → on

New test case:
yellow rubber cylinder

What is the detector light status?
Answer: on

Example 2:
Question: You are a researcher studying causal relationships using Blicket experiments. In these experiments, certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with:
- "on" if you can determine the detector light will turn on
- "off" if you can determine the detector light will stay off
- "undetermined" if there is insufficient evidence to reach a conclusion

Do not use quotation marks in your answer.

Previous experimental results:
yellow rubber cylinder → on
red rubber sphere → off
yellow rubber cylinder, red rubber sphere → on
yellow metal sphere, red metal cylinder, brown rubber cylinder, purple rubber sphere, yellow rubber cube → on
yellow rubber cube, brown rubber cylinder, purple rubber sphere → off
yellow metal sphere, red metal cylinder → on

New test case:
red metal cylinder

What is the detector light status?
Answer: undetermined

Example 3:
Question: You are a researcher studying causal relationships using Blicket experiments. In these experiments, certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with:
- "on" if you can determine the detector light will turn on
- "off" if you can determine the detector light will stay off
- "undetermined" if there is insufficient evidence to reach a conclusion

Do not use quotation marks in your answer.

Previous experimental results:
yellow rubber cylinder → on
red rubber sphere → off
yellow rubber cylinder, red rubber sphere → on
yellow metal sphere, red metal cylinder, brown rubber cylinder, purple rubber sphere, yellow rubber cube → on
yellow rubber cube, brown rubber cylinder, purple rubber sphere → off
yellow metal sphere, red metal cylinder → on

New test case:
yellow rubber cube, brown rubber cylinder, purple rubber sphere, red metal cylinder

What is the detector light status?
Answer: undetermined

````

### advanced_geometry
A dataset for advanced geometry tasks using coordinate geometry.

Default configuration:
```python
min_coord = -10
max_coord = 10
size = 50
seed = 42
task_types = ['orthocenter', 'incircle_radius', 'angle_measure']
```

Example tasks:
````
Example 1:
Question: In triangle ABC with coordinates A=(-7, -10), B=(-2, -3), and C=(-3, -6), find the measure (in degrees) of angle ABC. For all geometry problems:
1. Give coordinates in the form (x, y)
2. Round decimal answers to 3 decimal places
3. Use the degree symbol ° for angles
4. Return only the angle, coordinates, or radius as your answer.

Answer: 17.10°
Metadata: {'A': (-7, -10), 'B': (-2, -3), 'C': (-3, -6), 'angle_ABC_degrees': 17.10272896905237, 'task_type': 'angle_measure', 'difficulty': {'min_coord': -10, 'max_coord': 10}}

Example 2:
Question: For triangle with vertices A=(-1, -6), B=(4, 1), and C=(-7, 4), determine the orthocenter (intersection of altitudes). For all geometry problems:
1. Give coordinates in the form (x, y)
2. Round decimal answers to 3 decimal places
3. Use the degree symbol ° for angles
4. Return only the angle, coordinates, or radius as your answer.

Answer: (0.304, -1.217)
Metadata: {'A': (-1, -6), 'B': (4, 1), 'C': (-7, 4), 'ortho': (7/23, -28/23), 'orthocenter_exact': ('7/23', '-28/23'), 'orthocenter_approx': (0.30434782608695654, -1.2173913043478262), 'task_type': 'orthocenter', 'difficulty': {'min_coord': -10, 'max_coord': 10}}

Example 3:
Question: Find the incircle radius of triangle ABC whose vertices are A=(6, 7), B=(-7, -5), and C=(2, -3). For all geometry problems:
1. Give coordinates in the form (x, y)
2. Round decimal answers to 3 decimal places
3. Use the degree symbol ° for angles
4. Return only the angle, coordinates, or radius as your answer.

Answer: 2.176
Metadata: {'A': (6, 7), 'B': (-7, -5), 'C': (2, -3), 'incircle_radius_exact': 'sqrt(-sqrt(29) + sqrt(85)/2 + sqrt(313)/2)*sqrt(-sqrt(313)/2 + sqrt(85)/2 + sqrt(29))*sqrt(-sqrt(85)/2 + sqrt(29) + sqrt(313)/2)/sqrt(sqrt(85)/2 + sqrt(29) + sqrt(313)/2)', 'incircle_radius_approx': 2.176123777286009, 'task_type': 'incircle_radius', 'difficulty': {'min_coord': -10, 'max_coord': 10}}

````

### aiw
A procedural dataset inspired by the "Alice in Wonderland" paper.

    The dataset is inspired by the following paper:
       @inproceedings{nezhurina2024alice,
       title={Alice in Wonderland: Simple Tasks Reveal Severe Generalization and
              Basic Reasoning Deficits in State-Of-the-Art Large Language Models},
       author={Marianna Nezhurina and Lucia Cipolina-Kun and Mehdi Cherti and
              Jenia Jitsev},
       booktitle={NeurIPS 2024 Workshop on Scientific Methods for Understanding
                  Deep Learning},
       year={2024},
       url={https://openreview.net/forum?id=Mkl7dzjYiW}
       }

Default configuration:
```python
male_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Bob']
female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Margaret', 'Alice']
task_types = [<TaskType.SIBLINGS: 'siblings'>, <TaskType.FRIENDS: 'friends'>, <TaskType.COLLEAGUES: 'colleagues'>]
task_type_weights = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
seed = 42
size = 10
max_entities = 6
```

Example tasks:
````
Example 1:
Question: Mary has 2 male friends and she also has 2 female friends. They all are friends with each other and have no other friends aside. How many female friends does William, a male friend of Mary, have?
Answer: 3
Metadata: {'task_type': 'friends', 'difficulty': {'task_type_weight': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'num_entities': 6}}

Example 2:
Question: Jennifer has 3 brothers and she also has 6 sisters. How many sisters does Jennifer's brother have?
Answer: 7
Metadata: {'task_type': 'siblings', 'difficulty': {'task_type_weight': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'num_entities': 6}}

Example 3:
Question: Sarah has 2 male friends and she also has 4 female friends. They all are friends with each other and have no other friends aside. How many female friends does John, a male friend of Sarah, have?
Answer: 5
Metadata: {'task_type': 'friends', 'difficulty': {'task_type_weight': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'num_entities': 6}}

````

### arc_1d
Generates ARC 1D tasks by randomly selecting from available task generators

    This dataset is a procedural variant of the 1D-ARC dataset which is described in the paper:
    `LLMs and the Abstraction and Reasoning Corpus:  Successes, Failures, and the Importance
    of Object-based Representations` (https://arxiv.org/abs/2305.18354)

    Ilya Sheprut (optozorax) created rust generators for most of the ARC 1d tasks. For
    reasoning-gym rust tasks were machine-converted to python via Sonnet.

    Ilya's original rust code can be found here: https://github.com/optozorax/arc_1d/

Default configuration:
```python
min_size = 10
max_size = 30
num_train = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:
Input:  0 0 0 2 9 2 3 4 4 0
Output: 2 9 2 3 4 4 0 0 0 0

Example 2:
Input:  0 0 0 0 4 4 2 1 1 0
Output: 0 4 4 2 1 1 0 0 0 0

Example 3:
Input:  0 0 0 7 9 4 9 1 0 0
Output: 7 9 4 9 1 0 0 0 0 0

Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Your final answer should be just the test output grid itself.

Input:
0 0 0 0 0 1 5 0 0 0
Answer: 0 0 1 5 0 0 0 0 0 0
Metadata: {'task_name': 'move_3pix_colorful_left', 'size': 10, 'train_examples': [{'input': [0, 0, 0, 2, 9, 2, 3, 4, 4, 0], 'output': [2, 9, 2, 3, 4, 4, 0, 0, 0, 0]}, {'input': [0, 0, 0, 0, 4, 4, 2, 1, 1, 0], 'output': [0, 4, 4, 2, 1, 1, 0, 0, 0, 0]}, {'input': [0, 0, 0, 7, 9, 4, 9, 1, 0, 0], 'output': [7, 9, 4, 9, 1, 0, 0, 0, 0, 0]}], 'test_example': {'input': [0, 0, 0, 0, 0, 1, 5, 0, 0, 0], 'output': [0, 0, 1, 5, 0, 0, 0, 0, 0, 0]}}

Example 2:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:
Input:  0 0 0 0 0 0 0 6 2 8 8 1 0 0 0 0 0 0 0
Output: 0 0 0 0 0 0 0 0 6 2 8 8 1 0 0 0 0 0 0

Example 2:
Input:  0 6 9 7 7 3 1 2 2 7 3 2 3 9 8 3 7 9 0
Output: 0 0 6 9 7 7 3 1 2 2 7 3 2 3 9 8 3 7 9

Example 3:
Input:  0 0 0 0 0 0 0 0 0 3 7 2 1 1 3 1 3 5 0
Output: 0 0 0 0 0 0 0 0 0 0 3 7 2 1 1 3 1 3 5

Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Your final answer should be just the test output grid itself.

Input:
0 9 2 1 2 8 6 6 9 8 0 0 0 0 0 0 0 0 0
Answer: 0 0 9 2 1 2 8 6 6 9 8 0 0 0 0 0 0 0 0
Metadata: {'task_name': 'move_1pix_colorful_right', 'size': 19, 'train_examples': [{'input': [0, 0, 0, 0, 0, 0, 0, 6, 2, 8, 8, 1, 0, 0, 0, 0, 0, 0, 0], 'output': [0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 8, 8, 1, 0, 0, 0, 0, 0, 0]}, {'input': [0, 6, 9, 7, 7, 3, 1, 2, 2, 7, 3, 2, 3, 9, 8, 3, 7, 9, 0], 'output': [0, 0, 6, 9, 7, 7, 3, 1, 2, 2, 7, 3, 2, 3, 9, 8, 3, 7, 9]}, {'input': [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 2, 1, 1, 3, 1, 3, 5, 0], 'output': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 2, 1, 1, 3, 1, 3, 5]}], 'test_example': {'input': [0, 9, 2, 1, 2, 8, 6, 6, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'output': [0, 0, 9, 2, 1, 2, 8, 6, 6, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0]}}

Example 3:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:
Input:  0 0 0 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 0 0 0
Output: 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0

Example 2:
Input:  0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0
Output: 0 0 0 0 0 0 0 3 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0

Example 3:
Input:  5 5 5 5 5 5 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Output: 5 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. Your final answer should be just the test output grid itself.

Input:
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
Answer: 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0
Metadata: {'task_name': 'two_points_and_fill_inv', 'size': 26, 'train_examples': [{'input': [0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0], 'output': [0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0]}, {'input': [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'output': [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'input': [5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'output': [5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}], 'test_example': {'input': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0], 'output': [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]}}

````

### arc_agi
Default configuration:
```python
use_train = True
use_eval = True
board_format_opts = BoardFormattingOptions(alphabet=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], col_delimiter=' ', row_delimiter='\n', array_brackets=False)
rotations = ['90', '180', '270']
mirrors = ['horizontal', 'vertical', 'diagonal', 'counterdiagonal']
use_color_permutation = True
shuffle_example_order = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
7 7 7 7 7 6 3 6 7 7 7 6 6 7
7 7 7 7 7 6 6 6 7 7 7 6 6 7
6 6 6 6 7 6 6 6 7 7 7 6 6 7
6 3 6 6 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 6 6 3 6 7
7 7 7 7 7 7 7 7 7 6 3 6 6 7
7 7 7 6 6 6 6 7 7 6 6 6 3 7
7 7 7 6 6 3 6 7 7 7 7 7 7 7
7 7 7 6 3 6 6 7 7 7 7 7 7 7
7 7 7 6 6 6 6 7 7 7 6 3 6 6
7 7 7 7 7 7 7 7 7 7 6 6 6 3
7 7 7 7 7 7 7 7 7 7 6 3 3 6
7 7 7 7 7 7 7 7 7 7 6 6 6 6
Output:
7 7 7 7 7 6 3 6 7 7 7 6 6 7
7 7 7 7 7 6 6 6 7 7 7 6 6 7
6 6 6 6 7 6 6 6 7 7 7 6 6 7
6 3 6 6 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7

Example 2:

Input:
7 7 7 7 7 6 6 6 6 7 7 3 6 7 7
6 6 6 6 7 3 6 6 3 7 7 6 3 7 7
6 3 6 6 7 6 6 6 6 7 7 7 7 7 7
6 6 6 6 7 6 6 3 6 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6 6
7 7 7 7 7 7 7 7 7 7 7 6 6 6 6
7 7 6 6 3 6 6 7 7 7 7 7 7 7 7
7 7 6 6 6 3 6 7 7 7 7 7 7 7 7
7 7 6 3 6 6 6 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 6 6 3 7 7 7
7 7 6 6 6 6 7 7 7 6 3 6 7 7 7
7 7 6 6 6 6 7 7 7 6 6 6 7 7 7
7 7 6 6 6 6 7 7 7 3 6 3 7 7 7
Output:
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 7 7 7 7
6 3 6 6 7 7 7 7 7 7 7 7 7 7 7
6 6 6 6 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6 6
7 7 7 7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 6 6 6 6 7 7 7 7 7 7 7 7 7
7 7 6 6 6 6 7 7 7 7 7 7 7 7 7
7 7 6 6 6 6 7 7 7 7 7 7 7 7 7

Example 3:

Input:
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 6 3 6 7 7 7 7 7 7 7 7 7 7
7 6 6 3 7 6 6 6 7 7 6 3 7 7
7 7 7 7 7 6 3 6 7 7 6 6 7 7
7 7 7 7 7 6 6 3 7 7 7 7 7 7
7 7 7 7 7 3 6 6 7 7 7 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6
7 6 6 3 7 7 7 7 7 7 7 6 6 6
7 3 6 6 7 7 7 7 7 7 7 7 7 7
7 6 6 6 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 3 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
Output:
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 6 3 7 7
7 7 7 7 7 7 7 7 7 7 6 6 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 6 6 6
7 7 7 7 7 7 7 7 7 7 7 6 3 6
7 7 7 7 7 7 7 7 7 7 7 6 6 6
7 7 7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 3 6 6 7 7 7 7
7 7 7 7 7 7 7 6 6 6 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7 7 7


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
7 7 7 7 7 7 7 7 6 3 6 6
6 6 6 7 7 7 7 7 6 6 6 6
3 6 6 7 7 7 7 7 6 3 6 3
6 6 6 7 3 6 6 7 7 7 7 7
7 7 7 7 6 6 6 7 7 7 7 7
7 7 7 7 6 6 3 7 7 7 7 7
7 7 7 7 6 6 6 7 6 6 6 6
7 7 7 7 7 7 7 7 6 6 3 6
7 6 6 6 6 6 6 7 6 6 6 6
7 6 6 6 6 3 6 7 6 6 6 6
7 6 3 6 6 6 6 7 7 7 7 7
7 6 6 6 6 6 6 7 6 6 6 7
7 7 7 7 7 7 7 7 6 6 6 7

Answer: 7 7 7 7 7 7 7 7 7 7 7 7
6 6 6 7 7 7 7 7 7 7 7 7
3 6 6 7 7 7 7 7 7 7 7 7
6 6 6 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 6 6 3 6
7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 6 6 6 6
7 7 7 7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 6 6 6 7
7 7 7 7 7 7 7 7 6 6 6 7
Metadata: {'input': ((7, 7, 7, 7, 7, 7, 7, 7, 6, 3, 6, 6), (6, 6, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6), (3, 6, 6, 7, 7, 7, 7, 7, 6, 3, 6, 3), (6, 6, 6, 7, 3, 6, 6, 7, 7, 7, 7, 7), (7, 7, 7, 7, 6, 6, 6, 7, 7, 7, 7, 7), (7, 7, 7, 7, 6, 6, 3, 7, 7, 7, 7, 7), (7, 7, 7, 7, 6, 6, 6, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 3, 6), (7, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6), (7, 6, 6, 6, 6, 3, 6, 7, 6, 6, 6, 6), (7, 6, 3, 6, 6, 6, 6, 7, 7, 7, 7, 7), (7, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7)), 'output': ((7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7), (3, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7), (6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 3, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7)), 'task_id': 'a934301b'}

Example 2:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
2 8 8 8 8 8 8 8 8 9
2 8 8 0 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 0 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 0 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
Output:
2 8 8 8 8 8 8 8 8 9
2 8 8 2 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 9 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 9 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9
2 8 8 8 8 8 8 8 8 9

Example 2:

Input:
6 6 6 6 6 6 6 6 6 6
8 8 8 8 8 8 8 8 8 8
8 8 0 8 8 8 8 8 0 8
8 8 8 8 8 8 0 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 0 8 8 8 8
8 0 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
1 1 1 1 1 1 1 1 1 1
Output:
6 6 6 6 6 6 6 6 6 6
8 8 8 8 8 8 8 8 8 8
8 8 6 8 8 8 8 8 6 8
8 8 8 8 8 8 6 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 1 8 8 8 8
8 1 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
1 1 1 1 1 1 1 1 1 1

Example 3:

Input:
5 5 5 5 5 5 5 5 5 5
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 0 8 8 8 8
8 8 0 8 8 8 8 8 0 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 0 8 8 8 8 0 8
8 8 8 8 8 8 0 8 8 8
8 8 8 8 8 8 8 8 8 8
7 7 7 7 7 7 7 7 7 7
Output:
5 5 5 5 5 5 5 5 5 5
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 5 8 8 8 8
8 8 5 8 8 8 8 8 5 8
8 8 8 8 8 8 8 8 8 8
8 8 8 8 8 8 8 8 8 8
8 8 8 7 8 8 8 8 7 8
8 8 8 8 8 8 7 8 8 8
8 8 8 8 8 8 8 8 8 8
7 7 7 7 7 7 7 7 7 7


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
6 8 8 8 8 8 8 8 0 4
6 0 8 8 0 8 8 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 8 0 8 8 4
6 8 8 0 8 8 8 8 8 4
6 8 8 8 8 8 0 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 0 8 8 8 4
6 8 8 0 8 8 8 0 8 4
6 8 8 8 8 8 8 8 8 4

Answer: 6 8 8 8 8 8 8 8 4 4
6 6 8 8 6 8 8 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 8 4 8 8 4
6 8 8 6 8 8 8 8 8 4
6 8 8 8 8 8 4 8 8 4
6 8 8 8 8 8 8 8 8 4
6 8 8 8 8 4 8 8 8 4
6 8 8 6 8 8 8 4 8 4
6 8 8 8 8 8 8 8 8 4
Metadata: {'input': ((6, 8, 8, 8, 8, 8, 8, 8, 0, 4), (6, 0, 8, 8, 0, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 0, 8, 8, 4), (6, 8, 8, 0, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 0, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 0, 8, 8, 8, 4), (6, 8, 8, 0, 8, 8, 8, 0, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4)), 'output': ((6, 8, 8, 8, 8, 8, 8, 8, 4, 4), (6, 6, 8, 8, 6, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 4, 8, 8, 4), (6, 8, 8, 6, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 8, 4, 8, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4), (6, 8, 8, 8, 8, 4, 8, 8, 8, 4), (6, 8, 8, 6, 8, 8, 8, 4, 8, 4), (6, 8, 8, 8, 8, 8, 8, 8, 8, 4)), 'task_id': '2204b7a8'}

Example 3:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 5 5 5 5 5
5 5 8 8 8 8 5 5 5 5 5 8 8 8 8 5 5 5 5 5
2 5 8 8 8 8 5 5 5 5 5 8 8 8 8 5 5 5 5 2
5 5 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 8 8 8 8 8 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 5 5
Output:
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 8 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 8 5
5 5 8 8 8 8 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 8 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 2 2 2 2 5 5 5 5 5
5 5 2 2 2 2 5 2 5 5 5 2 2 2 2 5 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 5 2 2 2 2 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 2 2 2 2 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5 8 8 8 8 5 5

Example 2:

Input:
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 5 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 5 5 5 5 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 8 8 8 8 8 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 8 8 8 8 8 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 8 8 8 8 8 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
Output:
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 5 5 2 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 5 5 5 2 5 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 2 2 2 2 2 5 8 8 8 5 5 5 5 8 8 8 8
5 5 5 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 2 2 2 2 2 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 2 2 2 2 2 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5

Example 3:

Input:
5 8 8 8 8 8 5 2 5 5 5 5 5 5
5 8 8 8 8 8 5 5 5 5 5 8 8 8
5 5 5 5 5 5 5 5 5 5 5 8 8 8
5 5 5 5 8 8 8 8 8 8 5 8 8 8
5 5 5 5 8 8 8 8 8 8 5 8 8 8
5 5 5 5 8 8 8 8 8 8 5 8 8 8
8 8 5 5 8 8 8 8 8 8 5 5 5 5
8 8 5 5 8 8 8 8 8 8 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 8 8 8 5 5 8 8 8 5 5 5 5
2 5 8 8 8 5 5 8 8 8 5 5 5 2
5 5 8 8 8 5 5 5 5 5 5 5 5 5
5 5 8 8 8 5 5 2 5 5 5 5 5 5
Output:
5 8 8 8 8 8 5 2 5 5 5 5 5 5
5 8 8 8 8 8 5 2 5 5 5 8 8 8
5 5 5 5 5 5 5 2 5 5 5 8 8 8
5 5 5 5 2 2 2 2 2 2 5 8 8 8
5 5 5 5 2 2 2 2 2 2 5 8 8 8
5 5 5 5 2 2 2 2 2 2 5 8 8 8
8 8 5 5 2 2 2 2 2 2 5 5 5 5
8 8 5 5 2 2 2 2 2 2 5 5 5 5
5 5 5 5 5 5 5 2 5 5 5 5 5 5
5 5 2 2 2 5 5 2 2 2 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 5 2 2 2 5 5 2 5 5 5 5 5 5
5 5 2 2 2 5 5 2 5 5 5 5 5 5


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 8 8 5 5 5 5 5 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 8 8 5 5 8 8 8 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 8 8 8 8 8 8 8 8 8 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 8 8 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
2 8 8 8 8 8 5 5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 2
5 8 8 8 8 8 5 5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 8 8 5
5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 8 8 5
5 5 5 5 8 8 8 5 5 5 5 5 5 8 8 8 8 8 5 5 5 5 5 5 5
2 5 5 5 8 8 8 5 5 5 5 5 5 8 8 8 8 8 5 8 8 8 8 5 2
5 5 5 5 8 8 8 5 5 8 8 8 5 8 8 8 8 8 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 8 8 8 5 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 5 5 5 5 8 8 8 5 5 5 5 5 5 5 8 8 8 8 5 5
5 5 5 5 5 2 5 5 5 8 8 8 5 5 5 5 5 5 5 5 5 5 5 5 5

Answer: 5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 8 8 5 5 5 5 5 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 8 8 5 5 8 8 8 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 2 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 8 8 8 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 2 2 2 2 2 5 5 5 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 2 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 8 5
5 5 5 5 5 2 5 5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 8 8 5
5 5 5 5 5 2 5 5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 8 8 5
5 5 5 5 2 2 2 5 5 5 5 5 5 2 2 2 2 2 5 5 5 5 5 5 5
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
5 5 5 5 2 2 2 5 5 2 2 2 5 2 2 2 2 2 5 2 2 2 2 5 5
5 5 5 5 5 2 5 5 5 2 2 2 5 5 5 5 5 5 5 2 2 2 2 5 5
5 5 5 5 5 2 5 5 5 2 2 2 5 5 5 5 5 5 5 2 2 2 2 5 5
5 5 5 5 5 2 5 5 5 2 2 2 5 5 5 5 5 5 5 5 5 5 5 5 5
Metadata: {'input': ((5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (2, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2), (5, 8, 8, 8, 8, 8, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5), (2, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 5, 2), (5, 5, 5, 5, 8, 8, 8, 5, 5, 8, 8, 8, 5, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 5, 5), (5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)), 'output': ((5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 8, 8, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 8, 8, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), (5, 2, 2, 2, 2, 2, 5, 5, 5, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 8, 8, 5), (5, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), (5, 5, 5, 5, 2, 2, 2, 5, 5, 2, 2, 2, 5, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 5, 5), (5, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)), 'task_id': '0d87d2a6'}

````

### base_conversion
Generates base conversion tasks

Default configuration:
```python
min_base = 2
max_base = 16
min_value = 0
max_value = 1000
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to convert a number between two different bases.

If the target base is > 10, use lowercase letters a-z for digits above 9.

Now, convert the base-3 number 220020 to binary

Answer: 1010001110
Metadata: {'decimal_value': 654, 'source_base': 3, 'target_base': 2, 'source_repr': '220020', 'target_repr': '1010001110', 'difficulty': {'value': 654, 'base': (3, 2)}}

Example 2:
Question: Your task is to convert a number between two different bases.

If the target base is > 10, use lowercase letters a-z for digits above 9.

Now, convert the base-6 number 103 to base-13

Answer: 30
Metadata: {'decimal_value': 39, 'source_base': 6, 'target_base': 13, 'source_repr': '103', 'target_repr': '30', 'difficulty': {'value': 39, 'base': (6, 13)}}

Example 3:
Question: Your task is to convert a number between two different bases.

If the target base is > 10, use lowercase letters a-z for digits above 9.

Now, convert the base-10 number 418 to base-13

Answer: 262
Metadata: {'decimal_value': 418, 'source_base': 10, 'target_base': 13, 'source_repr': '418', 'target_repr': '262', 'difficulty': {'value': 418, 'base': (10, 13)}}

````

### basic_arithmetic
Dataset that generates basic arithmetic tasks with configurable complexity

Default configuration:
```python
min_terms = 2
max_terms = 6
min_digits = 1
max_digits = 4
operators = ('+', '-', '*', '/')
allow_parentheses = True
allow_negation = True
seed = 42
size = 500
format_style = simple
whitespace = single
```

Example tasks:
````
Example 1:
Question: Calculate -5 * -6.
Answer: 30
Metadata: {'expression': '-5 * -6', 'difficulty': {'num_terms': 2, 'num_digits': 1}}

Example 2:
Question: Calculate 965 / 5.
Answer: 193
Metadata: {'expression': '965 / 5', 'difficulty': {'num_terms': 2, 'num_digits': 3}}

Example 3:
Question: Calculate 0 + -2 + -4 * 0 * 3.
Answer: -2
Metadata: {'expression': '0 + -2 + -4 * 0 * 3', 'difficulty': {'num_terms': 5, 'num_digits': 1}}

````

### bf
Generates BF tasks

Default configuration:
```python
seed = 42
size = 500
difficulty = 1
```

Example tasks:
````
Example 1:
Question: This is a BF (Brainf*ck) computer program. What is the output?

>[-]>[-]<>++++++++++[<+++++++++++>-]<+.-.+++++.--------------.+++++++++++++++.<

Respond only with the exact output of the program.
Answer: onset
Metadata: {'bfit_code': '\nint main() {\n    print("onset");\n}\n', 'bf_program': '>[-]>[-]<>++++++++++[<+++++++++++>-]<+.-.+++++.--------------.+++++++++++++++.<', 'difficulty': {'difficulty': 1}}

Example 2:
Question: Consider the following BF (Brainf*ck) code. What would it output?

>[-]>[-]<>++++++++[<++++++++++++++>-]<.-----------.+++++++++++++.---------------.+++++.<

Provide only the exact output of the code.
Answer: perch
Metadata: {'bfit_code': '\nint main() {\n    print("perch");\n}\n', 'bf_program': '>[-]>[-]<>++++++++[<++++++++++++++>-]<.-----------.+++++++++++++.---------------.+++++.<', 'difficulty': {'difficulty': 1}}

Example 3:
Question: This is a BF (Brainf*ck) computer program. What is the output?

>[-]>[-]<>+++++++++[<+++++++++++++>-]<.-------.----------.+.+++++++++++++.<

Respond only with the exact output of the program.
Answer: under
Metadata: {'bfit_code': '\nint main() {\n    print("under");\n}\n', 'bf_program': '>[-]>[-]<>+++++++++[<+++++++++++++>-]<.-------.----------.+.+++++++++++++.<', 'difficulty': {'difficulty': 1}}

````

### binary_alternation
Generates Binary Alternation exercises with configurable difficulty

Default configuration:
```python
min_n = 10
max_n = 30
p_solvable = 0.8
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.

The string is called alternating if no two adjacent characters are equal. For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

Any two characters may be swapped, even if they are not adjacent.

Now, determine the minimum number of swaps to make the following binary string alternating: 011001010101010011101001010110

Answer: 5
Metadata: {'string': '011001010101010011101001010110', 'solution': 5, 'solvable': True, 'difficulty': {'n': 30}}

Example 2:
Question: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.

The string is called alternating if no two adjacent characters are equal. For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

Any two characters may be swapped, even if they are not adjacent.

Now, determine the minimum number of swaps to make the following binary string alternating: 00110110100

Answer: 3
Metadata: {'string': '00110110100', 'solution': 3, 'solvable': True, 'difficulty': {'n': 11}}

Example 3:
Question: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.

The string is called alternating if no two adjacent characters are equal. For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

Any two characters may be swapped, even if they are not adjacent.

Now, determine the minimum number of swaps to make the following binary string alternating: 00000001111110000111011

Answer: 5
Metadata: {'string': '00000001111110000111011', 'solution': 5, 'solvable': True, 'difficulty': {'n': 23}}

````

### binary_matrix
Generates Binary Matrix exercises with configurable difficulty

Default configuration:
```python
min_n = 3
max_n = 10
p_zero = 0.25
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

The output should be a matrix of the same size as the input matrix, where each cell contains the distance to the nearest 0.

Find the distance to the nearest 0 for each cell in the matrix below:
1 1 0 1
0 0 0 0
1 1 1 0
1 0 1 0

Answer: 1 1 0 1
0 0 0 0
1 1 1 0
1 0 1 0
Metadata: {'matrix': [[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0]], 'solution': [[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0]], 'difficulty': {'n': 4, 'p_zero': 0.25}}

Example 2:
Question: Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

The output should be a matrix of the same size as the input matrix, where each cell contains the distance to the nearest 0.

Find the distance to the nearest 0 for each cell in the matrix below:
1 0 1
1 1 1
1 0 1

Answer: 1 0 1
2 1 2
1 0 1
Metadata: {'matrix': [[1, 0, 1], [1, 1, 1], [1, 0, 1]], 'solution': [[1, 0, 1], [2, 1, 2], [1, 0, 1]], 'difficulty': {'n': 3, 'p_zero': 0.25}}

Example 3:
Question: Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

The output should be a matrix of the same size as the input matrix, where each cell contains the distance to the nearest 0.

Find the distance to the nearest 0 for each cell in the matrix below:
0 1 1 1 1 1 1 0 1
1 1 0 1 0 1 0 1 1
1 0 1 1 0 1 0 1 0
1 1 1 1 1 1 1 0 1
1 1 1 1 0 1 1 0 1
1 1 1 1 1 1 1 1 1
0 1 1 1 1 0 1 1 0
1 1 1 1 1 1 1 1 1
0 0 1 1 1 1 1 1 1

Answer: 0 1 1 2 1 2 1 0 1
1 1 0 1 0 1 0 1 1
1 0 1 1 0 1 0 1 0
2 1 2 2 1 2 1 0 1
2 2 2 1 0 1 1 0 1
1 2 3 2 1 1 2 1 1
0 1 2 2 1 0 1 1 0
1 1 2 3 2 1 2 2 1
0 0 1 2 3 2 3 3 2
Metadata: {'matrix': [[0, 1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1]], 'solution': [[0, 1, 1, 2, 1, 2, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 1, 0], [2, 1, 2, 2, 1, 2, 1, 0, 1], [2, 2, 2, 1, 0, 1, 1, 0, 1], [1, 2, 3, 2, 1, 1, 2, 1, 1], [0, 1, 2, 2, 1, 0, 1, 1, 0], [1, 1, 2, 3, 2, 1, 2, 2, 1], [0, 0, 1, 2, 3, 2, 3, 3, 2]], 'difficulty': {'n': 9, 'p_zero': 0.25}}

````

### bitwise_arithmetic
Dataset that generates tasks testing understanding of bitwise arithmetic operations.

    Generates expressions combining:
    - Standard arithmetic operators (+, -, *)
    - Bitwise shift operators (<<, >>)
    - Multi-byte hexadecimal numbers (e.g. 0x100 to 0xFFFF)

    The difficulty parameter controls expression complexity:
    - Level 1: Simple expressions like (0x123 + 0x456)
    - Level 2: Nested expressions with shifts like ((0x123 + 0x456) << 1)
    - Level 3+: Deeper nesting like ((0x123 + 0x456) << (0x789 >> 1))

    Each task provides:
    - A question asking to evaluate an expression
    - The correct answer in hexadecimal format
    - Metadata including the raw expression

    The dataset verifies answers by evaluating them as Python expressions,
    supporting both integer and hexadecimal string formats.

Default configuration:
```python
difficulty = 2
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. If the answer is negative, reply as a negative value (ex., -0x3), not the two's-compliment form. Reply only with the final hexidecimal value.
((0x3a24 - 0x24b8) + (0x1741 >> 0x3))
Answer: 0x1854
Metadata: {'problem': '((0x3a24 - 0x24b8) + (0x1741 >> 0x3))', 'difficulty': {'difficulty': 2}}

Example 2:
Question: Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. If the answer is negative, reply as a negative value (ex., -0x3), not the two's-compliment form. Reply only with the final hexidecimal value.
((0xacf1 * 0xb3cc) - (0x9a4b << 0x0))
Answer: 0x7975b8c1
Metadata: {'problem': '((0xacf1 * 0xb3cc) - (0x9a4b << 0x0))', 'difficulty': {'difficulty': 2}}

Example 3:
Question: Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. If the answer is negative, reply as a negative value (ex., -0x3), not the two's-compliment form. Reply only with the final hexidecimal value.
((0x2e39 + 0x622b) >> 0x0)
Answer: 0x9064
Metadata: {'problem': '((0x2e39 + 0x622b) >> 0x0)', 'difficulty': {'difficulty': 2}}

````

### boxnet
Default configuration:
```python
min_row_num = 1
max_row_num = 4
min_column_num = 2
max_column_num = 4
min_box_num = 1
max_box_num = 1
colour_list = ['red', 'blue', 'green']
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: 
You are a central planner tasked with directing agents in a grid-like field to move colored boxes to their corresponding color-coded targets.
Each agent occupies a 1x1 square and can only interact with objects within its square. Agents can move a box to an adjacent square or
directly to a target square of the same color. A square may contain multiple boxes and targets. The squares are identified by their center
coordinates (e.g., square[0.5, 0.5]). Actions are formatted as: move(box_color, destination), where box_color is the color of the box and
destination is either a target of the same color or an adjacent square. Your objective is to create a sequence of action plans that instructs
each agent to match all boxes to their color-coded targets in the most efficient manner.

Please adhere to the following rules when specifying your action plan:
1. Single Action per Agent: Assign only one action to each agent at a time. However, the final answer shoule be a list of action plans for multiple steps.
2. Unique Agent Keys: Use unique keys for each agent in the JSON format action plan. The key should be the agent's coordinates in the format "Agent[x, y]".
3. Prioritize Matching Boxes to Targets: Always prioritize actions that will match a box to its target over moving a box to an adjacent square.
4. Sequential Action Planning: The whole returned answer should be a list of action plans for multiple steps, do not just return one step plan.
5. Clear Formatting: Ensure the action plan is clearly formatted in JSON, with each agent's action specified as a key-value pair.
6. Conflict Resolution: Ensure that no two agents are assigned actions that would interfere with each other.
7. Optimize Efficiency: Aim to minimize the number of moves required to match all boxes with their targets.

Here is the format for your action plan:
Please provide your final answer as a list of action dictionaries.
For example:
```json
[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_red, target_red)"}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)", "Agent[2.5, 0.5]":"move...}, {...}...]
```
Include an agent in the action plan only if it has a task to perform next.


The current left boxes and agents are: Agent[0.5, 0.5]: I am in square[0.5, 0.5], I can observe ['box_red', 'target_red', 'box_blue', 'target_blue', 'box_green', 'target_green'], I can do ['move(box_red, square[0.5, 1.5])', 'move(box_red, target_red)', 'move(box_blue, square[0.5, 1.5])', 'move(box_blue, target_blue)', 'move(box_green, square[0.5, 1.5])', 'move(box_green, target_green)']
Agent[0.5, 1.5]: I am in square[0.5, 1.5], I can observe [], I can do []


Answer: None
Metadata: {'difficulty': {'row_num': 1, 'column_num': 2}, 'initial_state': {'0.5_0.5': ['box_red', 'target_red', 'box_blue', 'target_blue', 'box_green', 'target_green'], '0.5_1.5': []}}

Example 2:
Question: 
You are a central planner tasked with directing agents in a grid-like field to move colored boxes to their corresponding color-coded targets.
Each agent occupies a 1x1 square and can only interact with objects within its square. Agents can move a box to an adjacent square or
directly to a target square of the same color. A square may contain multiple boxes and targets. The squares are identified by their center
coordinates (e.g., square[0.5, 0.5]). Actions are formatted as: move(box_color, destination), where box_color is the color of the box and
destination is either a target of the same color or an adjacent square. Your objective is to create a sequence of action plans that instructs
each agent to match all boxes to their color-coded targets in the most efficient manner.

Please adhere to the following rules when specifying your action plan:
1. Single Action per Agent: Assign only one action to each agent at a time. However, the final answer shoule be a list of action plans for multiple steps.
2. Unique Agent Keys: Use unique keys for each agent in the JSON format action plan. The key should be the agent's coordinates in the format "Agent[x, y]".
3. Prioritize Matching Boxes to Targets: Always prioritize actions that will match a box to its target over moving a box to an adjacent square.
4. Sequential Action Planning: The whole returned answer should be a list of action plans for multiple steps, do not just return one step plan.
5. Clear Formatting: Ensure the action plan is clearly formatted in JSON, with each agent's action specified as a key-value pair.
6. Conflict Resolution: Ensure that no two agents are assigned actions that would interfere with each other.
7. Optimize Efficiency: Aim to minimize the number of moves required to match all boxes with their targets.

Here is the format for your action plan:
Please provide your final answer as a list of action dictionaries.
For example:
```json
[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_red, target_red)"}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)", "Agent[2.5, 0.5]":"move...}, {...}...]
```
Include an agent in the action plan only if it has a task to perform next.


The current left boxes and agents are: Agent[0.5, 0.5]: I am in square[0.5, 0.5], I can observe ['target_green'], I can do []
Agent[0.5, 1.5]: I am in square[0.5, 1.5], I can observe ['box_red', 'target_red', 'box_blue'], I can do ['move(box_red, square[0.5, 0.5])', 'move(box_red, square[0.5, 2.5])', 'move(box_red, target_red)', 'move(box_blue, square[0.5, 0.5])', 'move(box_blue, square[0.5, 2.5])']
Agent[0.5, 2.5]: I am in square[0.5, 2.5], I can observe ['target_blue', 'box_green'], I can do ['move(box_green, square[0.5, 1.5])']


Answer: None
Metadata: {'difficulty': {'row_num': 1, 'column_num': 3}, 'initial_state': {'0.5_0.5': ['target_green'], '0.5_1.5': ['box_red', 'target_red', 'box_blue'], '0.5_2.5': ['target_blue', 'box_green']}}

Example 3:
Question: 
You are a central planner tasked with directing agents in a grid-like field to move colored boxes to their corresponding color-coded targets.
Each agent occupies a 1x1 square and can only interact with objects within its square. Agents can move a box to an adjacent square or
directly to a target square of the same color. A square may contain multiple boxes and targets. The squares are identified by their center
coordinates (e.g., square[0.5, 0.5]). Actions are formatted as: move(box_color, destination), where box_color is the color of the box and
destination is either a target of the same color or an adjacent square. Your objective is to create a sequence of action plans that instructs
each agent to match all boxes to their color-coded targets in the most efficient manner.

Please adhere to the following rules when specifying your action plan:
1. Single Action per Agent: Assign only one action to each agent at a time. However, the final answer shoule be a list of action plans for multiple steps.
2. Unique Agent Keys: Use unique keys for each agent in the JSON format action plan. The key should be the agent's coordinates in the format "Agent[x, y]".
3. Prioritize Matching Boxes to Targets: Always prioritize actions that will match a box to its target over moving a box to an adjacent square.
4. Sequential Action Planning: The whole returned answer should be a list of action plans for multiple steps, do not just return one step plan.
5. Clear Formatting: Ensure the action plan is clearly formatted in JSON, with each agent's action specified as a key-value pair.
6. Conflict Resolution: Ensure that no two agents are assigned actions that would interfere with each other.
7. Optimize Efficiency: Aim to minimize the number of moves required to match all boxes with their targets.

Here is the format for your action plan:
Please provide your final answer as a list of action dictionaries.
For example:
```json
[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_red, target_red)"}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)", "Agent[2.5, 0.5]":"move...}, {...}...]
```
Include an agent in the action plan only if it has a task to perform next.


The current left boxes and agents are: Agent[0.5, 0.5]: I am in square[0.5, 0.5], I can observe ['target_blue', 'target_green'], I can do []
Agent[0.5, 1.5]: I am in square[0.5, 1.5], I can observe [], I can do []
Agent[0.5, 2.5]: I am in square[0.5, 2.5], I can observe [], I can do []
Agent[0.5, 3.5]: I am in square[0.5, 3.5], I can observe ['box_green'], I can do ['move(box_green, square[1.5, 3.5])', 'move(box_green, square[0.5, 2.5])']
Agent[1.5, 0.5]: I am in square[1.5, 0.5], I can observe [], I can do []
Agent[1.5, 1.5]: I am in square[1.5, 1.5], I can observe ['box_red'], I can do ['move(box_red, square[0.5, 1.5])', 'move(box_red, square[2.5, 1.5])', 'move(box_red, square[1.5, 0.5])', 'move(box_red, square[1.5, 2.5])']
Agent[1.5, 2.5]: I am in square[1.5, 2.5], I can observe [], I can do []
Agent[1.5, 3.5]: I am in square[1.5, 3.5], I can observe [], I can do []
Agent[2.5, 0.5]: I am in square[2.5, 0.5], I can observe [], I can do []
Agent[2.5, 1.5]: I am in square[2.5, 1.5], I can observe ['box_blue'], I can do ['move(box_blue, square[1.5, 1.5])', 'move(box_blue, square[3.5, 1.5])', 'move(box_blue, square[2.5, 0.5])', 'move(box_blue, square[2.5, 2.5])']
Agent[2.5, 2.5]: I am in square[2.5, 2.5], I can observe [], I can do []
Agent[2.5, 3.5]: I am in square[2.5, 3.5], I can observe [], I can do []
Agent[3.5, 0.5]: I am in square[3.5, 0.5], I can observe ['target_red'], I can do []
Agent[3.5, 1.5]: I am in square[3.5, 1.5], I can observe [], I can do []
Agent[3.5, 2.5]: I am in square[3.5, 2.5], I can observe [], I can do []
Agent[3.5, 3.5]: I am in square[3.5, 3.5], I can observe [], I can do []


Answer: None
Metadata: {'difficulty': {'row_num': 4, 'column_num': 4}, 'initial_state': {'0.5_0.5': ['target_blue', 'target_green'], '0.5_1.5': [], '0.5_2.5': [], '0.5_3.5': ['box_green'], '1.5_0.5': [], '1.5_1.5': ['box_red'], '1.5_2.5': [], '1.5_3.5': [], '2.5_0.5': [], '2.5_1.5': ['box_blue'], '2.5_2.5': [], '2.5_3.5': [], '3.5_0.5': ['target_red'], '3.5_1.5': [], '3.5_2.5': [], '3.5_3.5': []}}

````

### caesar_cipher
Generates Caesar cipher encryption/decryption tasks

Default configuration:
```python
delimiter = .
min_words = 3
max_words = 20
min_rotation = 1
max_rotation = 25
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Decrypt this Caesar cipher text: JNJUBUF ZPVS BTTPDJBUF XIPN J XBT DPNQMJNFOUJOH B NPNFOU BHP. Provide only the decrypted text as your final answer.
Answer: IMITATE YOUR ASSOCIATE WHOM I WAS COMPLIMENTING A MOMENT AGO
Metadata: {'rotation': 1, 'cipher_text': 'JNJUBUF ZPVS BTTPDJBUF XIPN J XBT DPNQMJNFOUJOH B NPNFOU BHP', 'clear_text': 'IMITATE YOUR ASSOCIATE WHOM I WAS COMPLIMENTING A MOMENT AGO', 'difficulty': {'rotation': 1, 'words': 10}}

Example 2:
Question: Decrypt this Caesar cipher text: PBSDJ XKZYVOYX CWSDR LYEQRD SD PYB K WOBO KXN YBSQSXKDON DOVOZRYXSM TYEBXKVSCW. Provide only the decrypted text as your final answer.
Answer: FRITZ NAPOLEON SMITH BOUGHT IT FOR A MERE AND ORIGINATED TELEPHONIC JOURNALISM
Metadata: {'rotation': 10, 'cipher_text': 'PBSDJ XKZYVOYX CWSDR LYEQRD SD PYB K WOBO KXN YBSQSXKDON DOVOZRYXSM TYEBXKVSCW', 'clear_text': 'FRITZ NAPOLEON SMITH BOUGHT IT FOR A MERE AND ORIGINATED TELEPHONIC JOURNALISM', 'difficulty': {'rotation': 10, 'words': 12}}

Example 3:
Question: Decrypt this Caesar cipher text: ZW PFLI JKFDRTY ZJ FLK FW ZK DLJK SV DVEUVU. Provide only the decrypted text as your final answer.
Answer: IF YOUR STOMACH IS OUT OF IT MUST BE MENDED
Metadata: {'rotation': 17, 'cipher_text': 'ZW PFLI JKFDRTY ZJ FLK FW ZK DLJK SV DVEUVU', 'clear_text': 'IF YOUR STOMACH IS OUT OF IT MUST BE MENDED', 'difficulty': {'rotation': 17, 'words': 10}}

````

### calendar_arithmetic
Default configuration:
```python
year = 2022
tasks = ['weekday_offset', 'weekday_of_date', 'weekday_of_date_from_first_date', 'recurring_event_day', 'count_days', 'count_business_days', 'is_leap_year']
offset_upper_bound = 100
leap_year_range = 200
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Between Sunday, February 27, 2022 and Wednesday, March 2, 2022 (counting both dates), what's the total count of business days (Monday through Friday)? Give the count numerically.
Answer: 3
Metadata: {'task': 'count_business_days', 'start_date': '2022-02-27', 'end_date': '2022-03-02', 'difficulty': {'task_complexity': 5, 'date_range': 100}}

Example 2:
Question: Starting from Monday, May 23, 2022, which weekday was it 98 days before? Write out the full weekday name.
Answer: Monday
Metadata: {'task': 'weekday_offset', 'start_date': '2022-05-23', 'offset_days': -98, 'target_date': '2022-02-14', 'difficulty': {'task_complexity': 0, 'date_range': 100}}

Example 3:
Question: If a meeting is scheduled on the last Saturday of September 2022, on which day of the month does it occur? Respond with just the number. Answer with -1 if the ordinal does not exist in the month.
Answer: 24
Metadata: {'task': 'recurring_event_day', 'year': 2022, 'month': 9, 'ordinal': 'last', 'weekday': 'Saturday', 'difficulty': {'task_complexity': 3, 'date_range': 100}}

````

### chain_sum
Generates simple arithmetic tasks using only + and - operators

Default configuration:
```python
min_terms = 2
max_terms = 6
min_digits = 1
max_digits = 4
allow_negation = False
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: State the final answer to the following arithmetic problem: 4 + 3 =
Answer: 7
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 1}, 'expression': '4 + 3'}

Example 2:
Question: State the final answer to the following arithmetic problem: 812 + 880 =
Answer: 1692
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 3}, 'expression': '812 + 880'}

Example 3:
Question: State the final answer to the following arithmetic problem: 2 + 6 + 3 + 4 + 0 =
Answer: 15
Metadata: {'difficulty': {'num_terms': 5, 'num_digits': 1}, 'expression': '2 + 6 + 3 + 4 + 0'}

````

### circuit_logic
Generates random digital logic circuits (in ASCII) together with:
      - a random Boolean expression,
      - random input assignments,
      - the final evaluated output.

    Each item in the dataset is a dict with:
       {
           "question": <str>,
           "answer": <str>,
           "metadata": {
               "diagram": <ASCII circuit diagram>,
               "expression": <str>,
               "term_strings": <list of term_strings>,
               "assignments": <dict of input->0/1>,
               "final_gate": <str>,
               "inputs": <list of input names>,
           }
       }

Default configuration:
```python
min_terms = 3
max_terms = 5
min_inputs = 2
max_inputs = 4
neg_prob = 0.3
allow_reuse = True
size = 100
seed = 42
```

Example tasks:
````
Example 1:
Question: Below is a randomly generated logic circuit.

A: ─────────────────┐
B: ───────────────┐ │
C: ─────────────┐ │ │
D: ───────────┐ │ │ │
E: ─────────┐ │ │ │ │
F: ───────┐ │ │ │ │ │
G: ─────┐ │ │ │ │ │ │
H: ───┐ │ │ │ │ │ │ │
I: ─┐ │ │ │ │ │ │ │ │
    │ │ │ │ │ │ │ │ ├─>o─│&&
    │ │ │ │ │ │ │ │ ├────│&&───┐
    │ │ │ │ │ │ │ ├───>o─│&&   │
    │ │ │ │ │ │ │ │ ├─>o─│&&   │
    │ │ │ │ │ │ │ │ │          │
    │ │ │ │ │ │ ├────────│&&   │
    │ │ │ │ │ ├──────────│&&──┐│
    │ │ │ │ │ └──────────│&&  ││
    │ │ │ │ ├─────────>o─│&&  ││
    │ │ │ │ │   │ │ │         │└───│++
    │ │ │ │ │   ├─────>o─│&&  └────│++
    │ │ │ └──────────────│&&───────│++─── OUT
    │ │ │   │   │ │ └────│&&  ┌────│++
    │ │ │   │   └────────│&&  │┌───│++
    │ │ │   │     │           ││
    │ │ └────────────────│⊕⊕  ││
    │ │     ├─────────>o─│⊕⊕──┘│
    │ ├──────────────────│⊕⊕   │
    │ │     │     │            │
    │ │     │     └───>o─│↑↑   │
    │ │     └────────────│↑↑───┘
    └────────────────────│↑↑
      └──────────────────│↑↑


Legend for gates:
&&: AND
↑↑: NAND
⊕⊕: XOR
>o: Negate
++: OR

Given the following input assignments:
  A = 1
  B = 0
  C = 1
  D = 1
  E = 0
  F = 1
  G = 0
  H = 0
  I = 0

What is the final output?
Answer: 1
Metadata: {'expression': "(A'&A&B'&A')+(C&D&D&E')+(C'&F&A&C)+(G⊕E'⊕H)+(B'↑E↑I↑H)", 'assignments': {'A': 1, 'B': 0, 'C': 1, 'D': 1, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 0}, 'term_strings': ["A'&A&B'&A'", "C&D&D&E'", "C'&F&A&C", "G⊕E'⊕H", "B'↑E↑I↑H"], 'final_gate': 'OR', 'inputs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'], 'difficulty': {'terms': 5, 'inputs': (2, 4)}}

Example 2:
Question: Below is a randomly generated logic circuit.

A: ───────┐
B: ─────┐ │
C: ───┐ │ │
D: ─┐ │ │ │
    │ │ │ ├────│⊕⊕──┐
    │ │ │ ├────│⊕⊕  │
    │ │ │ │         │
    │ │ │ ├────│↑↑  │
    │ │ ├──────│↑↑─┐└──│⊕⊕
    │ └────────│↑↑ └───│⊕⊕─── OUT
    │   │ ├────│↑↑  ┌──│⊕⊕
    │   │ │         │
    │   │ └────│&&  │
    │   └──────│&&──┘
    └──────────│&&


Legend for gates:
&&: AND
↑↑: NAND
⊕⊕: XOR
>o: Negate
⊕⊕: XOR

Given the following input assignments:
  A = 1
  B = 0
  C = 0
  D = 0

What is the final output?
Answer: 1
Metadata: {'expression': '(A⊕A)⊕(A↑B↑C↑A)⊕(A&B&D)', 'assignments': {'A': 1, 'B': 0, 'C': 0, 'D': 0}, 'term_strings': ['A⊕A', 'A↑B↑C↑A', 'A&B&D'], 'final_gate': 'XOR', 'inputs': ['A', 'B', 'C', 'D'], 'difficulty': {'terms': 3, 'inputs': (2, 4)}}

Example 3:
Question: Below is a randomly generated logic circuit.

A: ─────────┐
B: ───────┐ │
C: ─────┐ │ │
D: ───┐ │ │ │
E: ─┐ │ │ │ │
    │ │ │ │ ├─>o─│&&
    │ │ │ │ ├────│&&───┐
    │ │ │ │ ├────│&&   │
    │ │ │ │ │          │
    │ │ │ │ ├────│↑↑   │
    │ │ │ │ ├────│↑↑──┐│
    │ │ │ │ ├────│↑↑  │└──│++
    │ │ │ │ ├─>o─│↑↑  └───│++
    │ │ │ │ │        ┌────│++─── OUT
    │ │ │ │ ├─>o─│⊕⊕ │┌───│++
    │ │ │ │ ├─>o─│⊕⊕─┘│
    │ │ │ │ └────│⊕⊕  │
    │ │ │ │           │
    │ │ │ └──────│&&  │
    │ │ └────────│&&──┘
    │ └──────────│&&
    └────────────│&&


Legend for gates:
&&: AND
↑↑: NAND
⊕⊕: XOR
>o: Negate
++: OR

Given the following input assignments:
  A = 1
  B = 1
  C = 0
  D = 0
  E = 1

What is the final output?
Answer: 1
Metadata: {'expression': "(A'&A&A)+(A↑A↑A↑A')+(A'⊕A'⊕A)+(B&C&D&E)", 'assignments': {'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': 1}, 'term_strings': ["A'&A&A", "A↑A↑A↑A'", "A'⊕A'⊕A", 'B&C&D&E'], 'final_gate': 'OR', 'inputs': ['A', 'B', 'C', 'D', 'E'], 'difficulty': {'terms': 4, 'inputs': (2, 4)}}

````

### codeio
Exercise some caution when using this dataset, as it involves executing arbitrary code snippets.
    These code snippets are transformed by an LLM from raw code files which have been curated from high-quality sources.
    However, there is still a risk that the LLM could have introduced code with bad effects.

Default configuration:
```python
seed = 42
size = 500
input_prediction_probability = 0.5
```

Example tasks:
````
Example 1:
Question: 
You are given a question that requires some input and output variables as follows:

You are given an integer `number`. Your task is to determine if the number is a prime number. A prime number is a positive integer greater than 1 that has no positive divisors other than 1 and itself. The function should return a boolean value indicating whether the number is prime. The function should raise a `ValueError` if the input is not a positive integer.

The input and output requirements are as follows:

Input:
    number (int): A positive integer to check for primality.

Output:
    return (dict): A dictionary with one key:
    - is_prime (bool): A boolean value indicating whether the input number is prime.

Given the following input:

{'number': 114}

Can you predict the output without writing any code? Please think and then provide the exact output in the form of a JSON object as your final answer. The keys and values of the object should strictly match the output requirement as specified.

Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

def main_solution(number):
    import math

    if not isinstance(number, int) or not number >= 0:
        raise ValueError("is_prime() only accepts positive integers")

    if 1 < number < 4:
        return {"is_prime": True}
    elif number < 2 or number % 2 == 0 or number % 3 == 0:
        return {"is_prime": False}

    for i in range(5, int(math.sqrt(number) + 1), 6):
        if number % i == 0 or number % (i + 2) == 0:
            return {"is_prime": False}
    return {"is_prime": True}

Answer: {"is_prime": false}
Metadata: {'input_data': {'number': 114}, 'output_data': {'is_prime': False}}

Example 2:
Question: 
You are given a question that requires some input and output variables as follows:

You are given an integer `n`. Your task is to calculate the number of distinct terms in the sequence generated by `a^b` for all integers `a` and `b` where `2 <= a <= n` and `2 <= b <= n`. Return the count of distinct terms in the sequence.

The input and output requirements are as follows:

Input:
    n (int): An integer representing the upper limit for both `a` and `b` in the expression `a^b`.

Output:
    return (dict): A dictionary with one key:
    - distinct_terms (int): The number of distinct terms in the sequence generated by `a^b` for `2 <= a <= n` and `2 <= b <= n`.

Given the following output:

{'distinct_terms': 1224}

Can you predict a feasible input without writing any code? Please reason and put your final answer in the form of a JSON object, even if the there is only one input variable, with keys strictly matching the input variables' names as specified.

Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

def main_solution(n: int) -> dict:
    collect_powers = set()

    n = n + 1  # maximum limit

    for a in range(2, n):
        for b in range(2, n):
            current_pow = a**b  # calculates the current power
            collect_powers.add(current_pow)  # adds the result to the set
    return {"distinct_terms": len(collect_powers)}

Answer: {"n": 38}
Metadata: {'input_data': {'n': 38}, 'output_data': {'distinct_terms': 1224}}

Example 3:
Question: 
You are given a question that requires some input and output variables as follows:

You are given an integer `n`. Your task is to find the largest prime factor of `n`. A prime factor is a prime number that divides `n` without leaving a remainder. The function should return the largest prime factor of `n` as an integer. If `n` is less than or equal to 0, the function should raise a `ValueError`. If `n` is not an integer or cannot be cast to an integer, the function should raise a `TypeError`.

The input and output requirements are as follows:

Input:
    n (int): An integer greater than or equal to 1. The function will find the largest prime factor of this number.

Output:
    return (dict): A dictionary with one key:
    - largest_prime_factor (int): The largest prime factor of the input number `n`.

Given the following output:

{'largest_prime_factor': 2647}

Can you predict a feasible input without writing any code? Please reason and put your final answer in the form of a JSON object, even if the there is only one input variable, with keys strictly matching the input variables' names as specified.

Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

def main_solution(n: int) -> dict:
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise TypeError("Parameter n must be int or castable to int.")
    if n <= 0:
        raise ValueError("Parameter n must be greater than or equal to one.")
    i = 2
    ans = 0
    if n == 2:
        return {"largest_prime_factor": 2}
    while n > 2:
        while n % i != 0:
            i += 1
        ans = i
        while n % i == 0:
            n = n // i
        i += 1
    return {"largest_prime_factor": int(ans)}

Answer: {"n": 545282}
Metadata: {'input_data': {'n': 545282}, 'output_data': {'largest_prime_factor': 2647}}

````

### color_cube_rotation
Generates color cube rotation reasoning tasks

Default configuration:
```python
min_rotations = 1
max_rotations = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: A cube has:
- a pink top side
- a gray right side
- a orange front side
- a purple left side
- a indigo back side
- a cyan bottom side

The cube is rotated so that the side which was before at the bottom is now at the top.

What is now the color of the back side of the cube?
Provide only the color as your final answer.
Answer: orange
Metadata: {'initial_state': {'top': 'pink', 'right': 'gray', 'front': 'orange', 'left': 'purple', 'back': 'indigo', 'bottom': 'cyan'}, 'rotations': ['bottom'], 'target_side': 'back', 'num_rotations': 1, 'difficulty': {'rotations': 1}}

Example 2:
Question: A cube has:
- a gray top side
- a brown right side
- a silver front side
- a red left side
- a purple back side
- a yellow bottom side

The cube is rotated so that the side which was before at the left is now at the top.

Next, the bottom side is rotated to become the top face.

After that the cube is turned to make the bottom face the top.

What is now the color of the left side of the cube?
Provide only the color as your final answer.
Answer: yellow
Metadata: {'initial_state': {'top': 'gray', 'right': 'brown', 'front': 'silver', 'left': 'red', 'back': 'purple', 'bottom': 'yellow'}, 'rotations': ['left', 'bottom', 'bottom'], 'target_side': 'left', 'num_rotations': 3, 'difficulty': {'rotations': 3}}

Example 3:
Question: A cube has:
- a orange top side
- a cyan right side
- a violet front side
- a pink left side
- a gray back side
- a gold bottom side

The cube is rotated so that the side which was before at the left is now at the top.

Now the cube is rotated to place its back side at the top.

Now the cube is rotated to place its bottom side at the top.

What is now the color of the left side of the cube?
Provide only the color as your final answer.
Answer: gold
Metadata: {'initial_state': {'top': 'orange', 'right': 'cyan', 'front': 'violet', 'left': 'pink', 'back': 'gray', 'bottom': 'gold'}, 'rotations': ['left', 'back', 'bottom'], 'target_side': 'left', 'num_rotations': 3, 'difficulty': {'rotations': 3}}

````

### complex_arithmetic
Generates complex number arithmetic problems.

Default configuration:
```python
min_real = -10
max_real = 10
min_imag = -10
max_imag = 10
operations = ('+', '-', '*', '/')
operations_weights = [0.4, 0.4, 0.1, 0.1]
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Subtract the complex numbers: (-10.0 - 2.0i) - (-3.0 - 3.0i)
Answer: -7.0 + 1.0i
Metadata: {'num1': (-10.0, -2.0), 'num2': (-3.0, -3.0), 'operation': '-', 'result': (-7, 1), 'difficulty': {'min_real': -10, 'max_real': 10, 'min_imag': -10, 'max_imag': 10, 'operations_weights': [0.4, 0.4, 0.1, 0.1]}}

Example 2:
Question: Add the complex numbers: (-6.0 + 4.0i) + (1.0 - 7.0i)
Answer: -5.0 - 3.0i
Metadata: {'num1': (-6.0, 4.0), 'num2': (1.0, -7.0), 'operation': '+', 'result': (-5, -3), 'difficulty': {'min_real': -10, 'max_real': 10, 'min_imag': -10, 'max_imag': 10, 'operations_weights': [0.4, 0.4, 0.1, 0.1]}}

Example 3:
Question: Subtract the complex numbers: (7.0 - 7.0i) - (-5.0 + 2.0i)
Answer: 12.0 - 9.0i
Metadata: {'num1': (7.0, -7.0), 'num2': (-5.0, 2.0), 'operation': '-', 'result': (12, -9), 'difficulty': {'min_real': -10, 'max_real': 10, 'min_imag': -10, 'max_imag': 10, 'operations_weights': [0.4, 0.4, 0.1, 0.1]}}

````

### count_bits
Generates Count Bits exercises with configurable difficulty

Default configuration:
```python
min_n = 1
max_n = 2147483647
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: How many 1 bits are there in the binary representation of the number 1373158607?
Answer: 18
Metadata: {'number': 1373158607, 'solution': 18, 'binary': '1010001110110001011110011001111', 'difficulty': {'n': 1373158607}}

Example 2:
Question: How many 1 bits are there in the binary representation of the number 82789451?
Answer: 14
Metadata: {'number': 82789451, 'solution': 14, 'binary': '100111011110100010001001011', 'difficulty': {'n': 82789451}}

Example 3:
Question: How many 1 bits are there in the binary representation of the number 877324117?
Answer: 16
Metadata: {'number': 877324117, 'solution': 16, 'binary': '110100010010101110011101010101', 'difficulty': {'n': 877324117}}

````

### count_primes
Generates Count Primes exercises with configurable difficulty

Default configuration:
```python
min_n = 1
max_n = 10000
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Count how many prime numbers there are between 1825 and 2029 (inclusive) ?
Answer: 27
Metadata: {'start': 1825, 'end': 2029, 'primes': [1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029], 'solution': 27, 'difficulty': {'n': (1825, 2029)}}

Example 2:
Question: Count how many prime numbers there are between 632 and 5319 (inclusive) ?
Answer: 589
Metadata: {'start': 632, 'end': 5319, 'primes': [641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309], 'solution': 589, 'difficulty': {'n': (632, 5319)}}

Example 3:
Question: Count how many prime numbers there are between 6694 and 8824 (inclusive) ?
Answer: 236
Metadata: {'start': 6694, 'end': 8824, 'primes': [6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821], 'solution': 236, 'difficulty': {'n': (6694, 8824)}}

````

### countdown
Generates Countdown Number Game tasks

Default configuration:
```python
min_numbers = 4
max_numbers = 6
min_value = 1
max_value = 100
min_target = 100
max_target = 999
operators = ('+', '-', '*', '/')
shuffle = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Calculate 139 using all of these numbers: 36, 29, 95, 32, 4, 15.
Each number may be used at most once.

Final answer format instructions:
1. Provide your solution as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
5. Do not include any other text or formatting.

Answer: 15 - 4 + 95 + 36 - 32 + 29
Metadata: {'numbers': [36, 29, 95, 32, 4, 15], 'target': 139, 'expression': '15 - 4 + 95 + 36 - 32 + 29'}

Example 2:
Question: Using all the numbers 74, 48, 56, 66, create an expression that equals 132.
You can only use each number once.

Final answer format instructions:
1. Provide your solution as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
5. Do not include any other text or formatting.

Answer: 66 - 56 + 74 + 48
Metadata: {'numbers': [74, 48, 56, 66], 'target': 132, 'expression': '66 - 56 + 74 + 48'}

Example 3:
Question: Using all the numbers 5, 41, 38, 81, 14, create an expression that equals 450.
You can only use each number once.

Final answer format instructions:
1. Provide your solution as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
5. Do not include any other text or formatting.

Answer: 41*14 - 81 - 38 - 5
Metadata: {'numbers': [5, 41, 38, 81, 14], 'target': 450, 'expression': '41*14 - 81 - 38 - 5'}

````

### course_schedule
Generates Course Schedule exercises with configurable difficulty

Default configuration:
```python
min_num_courses = 5
max_num_courses = 10
min_num_prerequisites = 1
max_num_prerequisites = 2
min_cycle_length = 3
max_cycle_length = 5
p_solvable = 0.5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: There are a total of 10 courses you have to take, labeled from 0 to 9.

You are given the following list of prerequisites, where prerequisites[i] = (a_i, b_i) indicates that you must first take course b_i if you want to take course a_i:
[(9, 7), (8, 2), (7, 3), (8, 7), (1, 9), (1, 7), (2, 7), (5, 8), (9, 5), (2, 3), (6, 5), (3, 7), (4, 5), (3, 1), (0, 8), (5, 3)]

Return True if you can finish all courses considering the prerequisites, or False otherwise.

Answer: False
Metadata: {'courses': [7, 3, 2, 8, 5, 6, 9, 4, 0, 1], 'prerequisites': [(9, 7), (8, 2), (7, 3), (8, 7), (1, 9), (1, 7), (2, 7), (5, 8), (9, 5), (2, 3), (6, 5), (3, 7), (4, 5), (3, 1), (0, 8), (5, 3)], 'solution': False, 'solvable': False, 'difficulty': {'num_courses': 10}}

Example 2:
Question: There are a total of 5 courses you have to take, labeled from 0 to 4.

You are given the following list of prerequisites, where prerequisites[i] = (a_i, b_i) indicates that you must first take course b_i if you want to take course a_i:
[(1, 3), (0, 2), (1, 0), (3, 4), (3, 0), (4, 0), (2, 4), (2, 1)]

Return True if you can finish all courses considering the prerequisites, or False otherwise.

Answer: False
Metadata: {'courses': [0, 4, 3, 1, 2], 'prerequisites': [(1, 3), (0, 2), (1, 0), (3, 4), (3, 0), (4, 0), (2, 4), (2, 1)], 'solution': False, 'solvable': False, 'difficulty': {'num_courses': 5}}

Example 3:
Question: There are a total of 8 courses you have to take, labeled from 0 to 7.

You are given the following list of prerequisites, where prerequisites[i] = (a_i, b_i) indicates that you must first take course b_i if you want to take course a_i:
[(6, 0), (2, 5), (3, 2), (3, 6), (6, 4), (5, 4), (2, 4), (1, 4), (0, 4), (7, 6)]

Return True if you can finish all courses considering the prerequisites, or False otherwise.

Answer: True
Metadata: {'courses': [4, 5, 0, 2, 6, 3, 7, 1], 'prerequisites': [(6, 0), (2, 5), (3, 2), (3, 6), (6, 4), (5, 4), (2, 4), (1, 4), (0, 4), (7, 6)], 'solution': True, 'solvable': True, 'difficulty': {'num_courses': 8}}

````

### cryptarithm
Generates cryptarithm puzzles by:
      1) Randomly choosing integers for each "addend" (with no leading zero if not allowed),
      2) Summing them,
      3) Mapping distinct digits (0..9) to letters (A..Z),
      4) Formatting the puzzle text.

    This approach guarantees sum correctness and avoids repeated failures.

Default configuration:
```python
min_words = 2
max_words = 3
allow_leading_zero = False
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve this cryptarithm:

    FOM
+ IKPLO
-------
  IKIZL

Each letter stands for a unique digit (0-9). No leading letter can be zero.
Provide a comma separated mapping from letters to digits that satisfies the equation in your final answer. Output format: "A=1,B=2,C=3" (without quotes)

Answer: F=3,I=4,K=2,L=9,M=1,O=8,P=0,Z=7
Metadata: {'letters': ['L', 'O', 'K', 'I', 'P', 'Z', 'M', 'F'], 'word_values': [381, 42098], 'sum_number': 42479, 'words_letters': ['FOM', 'IKPLO'], 'result_letters': 'IKIZL', 'digit_to_letter': {'9': 'L', '8': 'O', '2': 'K', '4': 'I', '0': 'P', '7': 'Z', '1': 'M', '3': 'F'}, 'letter_to_digit': {'L': 9, 'O': 8, 'K': 2, 'I': 4, 'P': 0, 'Z': 7, 'M': 1, 'F': 3}, 'difficulty': {'min_words': 2, 'max_words': 3}}

Example 2:
Question: Solve this cryptarithm:

   HHPD
+ JIOKP
-------
  JHEDH

Each letter stands for a unique digit (0-9). No leading letter can be zero.
Provide a comma separated mapping from letters to digits that satisfies the equation in your final answer. Output format: "A=1,B=2,C=3" (without quotes)

Answer: D=8,E=9,H=3,I=0,J=7,K=2,O=6,P=5
Metadata: {'letters': ['O', 'K', 'H', 'P', 'I', 'D', 'E', 'J'], 'word_values': [3358, 70625], 'sum_number': 73983, 'words_letters': ['HHPD', 'JIOKP'], 'result_letters': 'JHEDH', 'digit_to_letter': {'6': 'O', '2': 'K', '3': 'H', '5': 'P', '0': 'I', '8': 'D', '9': 'E', '7': 'J'}, 'letter_to_digit': {'O': 6, 'K': 2, 'H': 3, 'P': 5, 'I': 0, 'D': 8, 'E': 9, 'J': 7}, 'difficulty': {'min_words': 2, 'max_words': 3}}

Example 3:
Question: Solve this cryptarithm:

   RZRHA
   PPXZZ
+  ZHGZA
--------
  XXNXHZ

Each letter stands for a unique digit (0-9). No leading letter can be zero.
Provide a comma separated mapping from letters to digits that satisfies the equation in your final answer. Output format: "A=1,B=2,C=3" (without quotes)

Answer: A=0,G=7,H=9,N=8,P=3,R=2,X=1,Z=5
Metadata: {'letters': ['Z', 'H', 'N', 'G', 'X', 'A', 'R', 'P'], 'word_values': [25290, 33155, 59750], 'sum_number': 118195, 'words_letters': ['RZRHA', 'PPXZZ', 'ZHGZA'], 'result_letters': 'XXNXHZ', 'digit_to_letter': {'5': 'Z', '9': 'H', '8': 'N', '7': 'G', '1': 'X', '0': 'A', '2': 'R', '3': 'P'}, 'letter_to_digit': {'Z': 5, 'H': 9, 'N': 8, 'G': 7, 'X': 1, 'A': 0, 'R': 2, 'P': 3}, 'difficulty': {'min_words': 2, 'max_words': 3}}

````

### decimal_arithmetic
Dataset that generates basic arithmetic tasks using Decimal arithmetic and proper operator precedence.

Default configuration:
```python
min_num_decimal_places = 3
max_num_decimal_places = 3
min_terms = 2
max_terms = 6
precision = 12
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please solve this problem to a maximum of 12 significant digits, rounding up from the half. Only reply with the final value.
(4.507-2.287) = ?
Answer: 2.220
Metadata: {'difficulty': {'decimal_places': 3, 'num_terms': 2}}

Example 2:
Question: Please solve this problem to a maximum of 12 significant digits, rounding up from the half. Only reply with the final value.
2.359/1.578 = ?
Answer: 1.49493029151
Metadata: {'difficulty': {'decimal_places': 3, 'num_terms': 2}}

Example 3:
Question: Please solve this problem to a maximum of 12 significant digits, rounding up from the half. Only reply with the final value.
((2.895/4.745)+1.912+2.568*9.683) = ?
Answer: 27.3880599115
Metadata: {'difficulty': {'decimal_places': 3, 'num_terms': 5}}

````

### decimal_chain_sum
Generates simple decimal arithmetic tasks using only + and - operators

Default configuration:
```python
min_terms = 2
max_terms = 6
min_digits = 1
max_digits = 4
min_decimal_places = 1
max_decimal_places = 4
allow_negation = False
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: State the final answer to the following arithmetic problem: 4.23 + 3.96 =
Answer: 8.19
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 1}, 'expression': '4.23 + 3.96'}

Example 2:
Question: State the final answer to the following arithmetic problem: 812.57 - 880.2577 =
Answer: -67.6877
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 3}, 'expression': '812.57 - 880.2577'}

Example 3:
Question: State the final answer to the following arithmetic problem: 2.75 - 6.5 - 3.7 + 4.7 - 0.98 =
Answer: -3.73
Metadata: {'difficulty': {'num_terms': 5, 'num_digits': 1}, 'expression': '2.75 - 6.5 - 3.7 + 4.7 - 0.98'}

````

### dice
Generates Dice-based puzzles with configurable parameters

Default configuration:
```python
num_dice = 4
max_dice_size = 20
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: I have these dice: 1d20, 1d10, 1d5, 1d2. What are the odds of rolling 18 or higher? (Assume that all dice are rolled at once, and that '1d6' represents one roll of a 6-sided dice.) Please respond with a reduced fraction representing the probability [ex., 1/60].
Answer: 13/20
Metadata: {'difficulty': {'num_dice': 4, 'max_dice_size': 20}, 'puzzle': {'dice_str': '1d20, 1d10, 1d5, 1d2', 'target': 18, 'num': 13, 'den': 20, 'probability': 0.65}}

Example 2:
Question: I have these dice: 1d20, 1d11, 1d6, 1d3. What are the odds of rolling 23 or higher? (Assume that all dice are rolled at once, and that '1d6' represents one roll of a 6-sided dice.) Please respond with a reduced fraction representing the probability [ex., 1/60].
Answer: 19/40
Metadata: {'difficulty': {'num_dice': 4, 'max_dice_size': 20}, 'puzzle': {'dice_str': '1d20, 1d11, 1d6, 1d3', 'target': 23, 'num': 19, 'den': 40, 'probability': 0.475}}

Example 3:
Question: I have these dice: 1d20, 1d19, 1d18, 1d15. What are the odds of rolling 48 or higher? (Assume that all dice are rolled at once, and that '1d6' represents one roll of a 6-sided dice.) Please respond with a reduced fraction representing the probability [ex., 1/60].
Answer: 9677/51300
Metadata: {'difficulty': {'num_dice': 4, 'max_dice_size': 20}, 'puzzle': {'dice_str': '1d20, 1d19, 1d18, 1d15', 'target': 48, 'num': 9677, 'den': 51300, 'probability': 0.18863547758284602}}

````

### emoji_mystery
Default configuration:
```python
size = 1000
seed = 42
min_words_in_sentence = 3
max_words_in_sentence = 35
```

Example tasks:
````
Example 1:
Question: The following emoji is encoded with a sentence.

Decode the following sentence from the emoji: 😽󠄶󠅢󠅙󠅤󠅪󠄐󠄾󠅑󠅠󠅟󠅜󠅕󠅟󠅞󠄐󠅃󠅝󠅙󠅤󠅘󠄐󠅑󠅧󠅟󠅛󠅕󠄐󠅙󠅞󠄐󠅦󠅕󠅢󠅩󠄐󠅒󠅑󠅔󠄐󠅘󠅥󠅝󠅟󠅢󠄞

Here is a hint:
```python
def variance_selector_to_byte(variation_selector):
    variation_selector_codepoint = ord(variation_selector)
    if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
        return variation_selector_codepoint - 0xFE00
    elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
        return variation_selector_codepoint - 0xE0100 + 16
    else:
        return None

def decode(encoded_sentence):
    decoded_bytes = []
    variation_selectors_part = encoded_sentence[1:]
    for char in variation_selectors_part:
        byte_val = variance_selector_to_byte(char)
        if byte_val is not None:
            decoded_bytes.append(byte_val)
    return bytes(decoded_bytes).decode('utf-8')
```


Return the secret sentence as your final answer.

Answer: Fritz Napoleon Smith awoke in very bad humor.
Metadata: {'emoji': '😽', 'difficulty': {'num_words_in_sentence': 8}}

Example 2:
Question: The following emoji is encoded with a sentence.

Decode the following sentence from the emoji: 😆󠄱󠅞󠅔󠄐󠅙󠅞󠅔󠅕󠅕󠅔󠄜󠄐󠅣󠅟󠄐󠅓󠅟󠅞󠅖󠅙󠅔󠅕󠅞󠅤󠄐󠅧󠅑󠅣󠄐󠅤󠅘󠅕󠅙󠅢󠄐󠅑󠅙󠅢󠄜󠄐󠅤󠅘󠅕󠅩󠄐󠅣󠅕󠅕󠅝󠅕󠅔󠄐󠅤󠅟󠄐󠅘󠅑󠅦󠅕󠄐󠅞󠅟󠄐󠅔󠅟󠅥󠅒󠅤󠄐󠅑󠅣󠄐󠅤󠅟󠄐󠅤󠅘󠅕󠄐󠅠󠅟󠅣󠅣󠅙󠅒󠅙󠅜󠅙󠅤󠅩󠄐󠅟󠅖󠄐󠅣󠅥󠅓󠅓󠅕󠅣󠅣󠄐󠅙󠅞󠄐󠅣󠅥󠅓󠅘󠄐󠅑󠅞󠄐󠅥󠅞󠅔󠅕󠅢󠅤󠅑󠅛󠅙󠅞󠅗󠄞

Here is a hint:
```python
def variance_selector_to_byte(variation_selector):
    variation_selector_codepoint = ord(variation_selector)
    if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
        return variation_selector_codepoint - 0xFE00
    elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
        return variation_selector_codepoint - 0xE0100 + 16
    else:
        return None

def decode(encoded_sentence):
    decoded_bytes = []
    variation_selectors_part = encoded_sentence[1:]
    for char in variation_selectors_part:
        byte_val = variance_selector_to_byte(char)
        if byte_val is not None:
            decoded_bytes.append(byte_val)
    return bytes(decoded_bytes).decode('utf-8')
```


Return the secret sentence as your final answer.

Answer: And indeed, so confident was their air, they seemed to have no doubt as to the possibility of success in such an undertaking.
Metadata: {'emoji': '😆', 'difficulty': {'num_words_in_sentence': 23}}

Example 3:
Question: The following emoji is encoded with a sentence.

Decode the following sentence from the emoji: 😱󠄹󠅞󠄐󠅖󠅑󠅓󠅤󠄜󠄐󠅟󠅞󠅕󠄐󠅟󠅖󠄐󠅤󠅘󠅕󠅝󠄐󠅧󠅑󠅞󠅤󠅕󠅔󠄐󠅤󠅟󠄐󠅢󠅕󠅦󠅙󠅦󠅕󠄐󠅠󠅑󠅙󠅞󠅤󠅙󠅞󠅗󠄜󠄐󠅑󠅞󠄐󠅑󠅢󠅤󠄐󠅖󠅑󠅜󠅜󠅕󠅞󠄐󠅙󠅞󠅤󠅟󠄐󠅔󠅕󠅣󠅥󠅕󠅤󠅥󠅔󠅕󠄐󠅟󠅧󠅙󠅞󠅗󠄐󠅤󠅟󠄐󠅤󠅘󠅕󠄐󠅠󠅢󠅟󠅗󠅢󠅕󠅣󠅣󠄐󠅝󠅑󠅔󠅕󠄐󠅙󠅞󠄐󠅓󠅟󠅜󠅟󠅢󠄝󠅠󠅘󠅟󠅤󠅟󠅗󠅢󠅑󠅠󠅘󠅩󠄞

Here is a hint:
```python
def variance_selector_to_byte(variation_selector):
    variation_selector_codepoint = ord(variation_selector)
    if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
        return variation_selector_codepoint - 0xFE00
    elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
        return variation_selector_codepoint - 0xE0100 + 16
    else:
        return None

def decode(encoded_sentence):
    decoded_bytes = []
    variation_selectors_part = encoded_sentence[1:]
    for char in variation_selectors_part:
        byte_val = variance_selector_to_byte(char)
        if byte_val is not None:
            decoded_bytes.append(byte_val)
    return bytes(decoded_bytes).decode('utf-8')
```


Return the secret sentence as your final answer.

Answer: In fact, one of them wanted to revive painting, an art fallen into desuetude owing to the progress made in color-photography.
Metadata: {'emoji': '😱', 'difficulty': {'num_words_in_sentence': 22}}

````

### family_relationships
Generates family relationship reasoning tasks

Default configuration:
```python
min_family_size = 4
max_family_size = 8
male_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Peter', 'Daniel', 'Matthew', 'Christopher', 'Andrew', 'George', 'Edward', 'Benjamin', 'Henry', 'Samuel', 'Alexander', 'Oliver', 'Jack', 'Harry', 'Jacob', 'Noah', 'Ethan', 'Lucas', 'Mason', 'Logan', 'Sebastian', 'Theodore', 'Owen', 'Liam', 'Aiden', 'Kai', 'Jayden', 'Zion', 'Phoenix', 'Atlas', 'Axel', 'Ryder', 'Finn']
female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Emma', 'Lisa', 'Anna', 'Margaret', 'Victoria', 'Charlotte', 'Sophia', 'Isabella', 'Olivia', 'Ava', 'Mia', 'Emily', 'Abigail', 'Amelia', 'Eleanor', 'Grace', 'Alice', 'Lucy', 'Chloe', 'Sophie', 'Lily', 'Hannah', 'Zoe', 'Luna', 'Nova', 'Aria', 'Willow', 'Aurora', 'Sage', 'River', 'Winter', 'Sky', 'Rain']
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: John is married to Isabella. They have a child called Edward. Edward is married to Victoria.

What is Isabella to Edward? Respond only with the word that describes their relationship.
Answer: mother
Metadata: {'person1': 'Isabella', 'person2': 'Edward', 'relationship': 'mother', 'family_size': 4, 'difficulty': {'family_size': 4}}

Example 2:
Question: Henry is married to Karen. They have a child called Sebastian. Sebastian is married to Eleanor.

What relation is Henry to Karen? Answer with a single word.
Answer: husband
Metadata: {'person1': 'Henry', 'person2': 'Karen', 'relationship': 'husband', 'family_size': 4, 'difficulty': {'family_size': 4}}

Example 3:
Question: Liam is married to Nova. They have a child called Noah. Noah is married to Charlotte. They have a child called Patricia. Joseph is married to Lisa. They have a child called Charlotte.

What is Liam to Noah? Respond only with the word that describes their relationship.
Answer: father
Metadata: {'person1': 'Liam', 'person2': 'Noah', 'relationship': 'father', 'family_size': 7, 'difficulty': {'family_size': 7}}

````

### figlet_font
Generates FigletFont tasks

Default configuration:
```python
static_word = None
static_font = None
space_letters = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please read the following figlet font:

  # #    ##   ##   ####   ## ###  ######   
 ## ##   ###  ##  ######  ## ###  ######   
### ###   ### ##  ##  ##  ##               
##   ##  # #####   ##     ## ###    ##     
##   ##  ## ####     ##   ## ###    ##     
### ###  ##  ###  ##  ##  ##        ##     
 ## ##   ##   ##  ######  ## ###    ##     
  # #    ##    #   ####   ## ###    ##     

Answer: ONSET
Metadata: {'font': 'pawn_ins', 'space_letters': True}

Example 2:
Question: What word does this say?

 888888ba      88888888b     888888ba      a88888b.    dP     dP  
 88    `8b     88            88    `8b    d8'   `88    88     88  
a88aaaa8P'    a88aaaa       a88aaaa8P'    88           88aaaaa88a 
 88            88            88   `8b.    88           88     88  
 88            88            88     88    Y8.   .88    88     88  
 dP            88888888P     dP     dP     Y88888P'    dP     dP  
                                                                  
                                                                  

Answer: PERCH
Metadata: {'font': 'nancyj-improved', 'space_letters': True}

Example 3:
Question: What word does this say?

##   ##  ###  ##  ######   #######  ######   
##   ##   ### ##   ## ###   ##  ##   ##  ##  
##   ##   ######   ##  ##   ##       ##  ##  
##   ##   ## ###   ##  ##   ####     #####   
##   ##   ##  ##   ##  ##   ##       ##  ##  
### ###   ##  ##   ## ###   ##  ##   ##  ##  
 #####   ###  ##  ######   #######  #### ### 
                                             

Answer: UNDER
Metadata: {'font': 'fp2_____', 'space_letters': True}

````

### fraction_simplification
Generates fraction simplification tasks

Default configuration:
```python
min_value = 1
max_value = 1000
min_factor = 1
max_factor = 100
styles = ('plain', 'latex_inline', 'latex_frac', 'latex_dfrac')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Simplify the fraction $\frac{92}{524}$ to its lowest terms. Give only the simplified fraction as your final answer.
Answer: $\frac{23}{131}$
Metadata: {'numerator': 92, 'denominator': 524, 'simplified_numerator': 23, 'simplified_denominator': 131, 'reduction_factor': 4, 'style': 'latex_frac', 'difficulty': {'factor': 4, 'value': (23, 131)}}

Example 2:
Question: Simplify the fraction $3600/26370$ to its lowest terms. Give only the simplified fraction as your final answer.
Answer: $40/293$
Metadata: {'numerator': 3600, 'denominator': 26370, 'simplified_numerator': 40, 'simplified_denominator': 293, 'reduction_factor': 90, 'style': 'latex_inline', 'difficulty': {'factor': 90, 'value': (40, 293)}}

Example 3:
Question: Simplify the fraction 29330/37310 to its lowest terms. Give only the simplified fraction as your final answer.
Answer: 419/533
Metadata: {'numerator': 29330, 'denominator': 37310, 'simplified_numerator': 419, 'simplified_denominator': 533, 'reduction_factor': 70, 'style': 'plain', 'difficulty': {'factor': 70, 'value': (419, 533)}}

````

### futoshiki
Generates Futoshiki puzzles with configurable board size and difficulty

Default configuration:
```python
min_board_size = 4
max_board_size = 9
min_difficulty = 0
max_difficulty = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve the following 9x9 Futoshiki puzzle:

8   4   _   9   _   _   2   5   _
                                 
_   _   5   3   2   _   _   1   _
                                 
3   _   _   8   _   _   _   _   _
                            ∧    
_   _   6   4   3   2   _   _   5
                                 
_   _   2   7   _   1   9   6   4
        ∧   ∨                    
7   9   _   _   1   4   5   _   _
                                 
_   _   _   6   _   5   3   9   8
                                 
_   _   _   _   _   _   4   2   _
                                 
_   7   9   _   _   _   _   8   2

Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.
Use < and > for horizontal constraints. Use ∧ and ∨ for vertical constraints.
Remember, in Futoshiki each row and column must contain each number from 1 to 9 exactly once.
Answer: 8   4   3   9   6   7   2   5   1
                                 
4   6   5   3   2   9   8   1   7
                                 
3   2   1   8   5   6   7   4   9
                            ∧    
9   8   6   4   3   2   1   7   5
                                 
5   3   2   7   8   1   9   6   4
        ∧   ∨                    
7   9   8   2   1   4   5   3   6
                                 
2   1   4   6   7   5   3   9   8
                                 
6   5   7   1   9   8   4   2   3
                                 
1   7   9   5   4   3   6   8   2
Metadata: {'puzzle': [[8, 4, 0, 9, 0, 0, 2, 5, 0], [0, 0, 5, 3, 2, 0, 0, 1, 0], [3, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 6, 4, 3, 2, 0, 0, 5], [0, 0, 2, 7, 0, 1, 9, 6, 4], [7, 9, 0, 0, 1, 4, 5, 0, 0], [0, 0, 0, 6, 0, 5, 3, 9, 8], [0, 0, 0, 0, 0, 0, 4, 2, 0], [0, 7, 9, 0, 0, 0, 0, 8, 2]], 'constraints': {((2, 7), (3, 7)): '<', ((4, 2), (5, 2)): '<', ((4, 3), (5, 3)): '>'}, 'solution': [[8, 4, 3, 9, 6, 7, 2, 5, 1], [4, 6, 5, 3, 2, 9, 8, 1, 7], [3, 2, 1, 8, 5, 6, 7, 4, 9], [9, 8, 6, 4, 3, 2, 1, 7, 5], [5, 3, 2, 7, 8, 1, 9, 6, 4], [7, 9, 8, 2, 1, 4, 5, 3, 6], [2, 1, 4, 6, 7, 5, 3, 9, 8], [6, 5, 7, 1, 9, 8, 4, 2, 3], [1, 7, 9, 5, 4, 3, 6, 8, 2]], 'difficulty': {'board_size': 9, 'difficulty': 0}}

Example 2:
Question: Solve the following 4x4 Futoshiki puzzle:

_   3 < _ > _
∧   ∨   ∨    
_ > _   _   _
        ∨    
_   _ > _ < _
∧            
_   1   _   _

Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.
Use < and > for horizontal constraints. Use ∧ and ∨ for vertical constraints.
Remember, in Futoshiki each row and column must contain each number from 1 to 4 exactly once.
Answer: 1   3 < 4 > 2
∧   ∨   ∨    
4 > 2   3   1
        ∨    
2   4 > 1 < 3
∧            
3   1   2   4
Metadata: {'puzzle': [[0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], 'constraints': {((0, 0), (1, 0)): '<', ((0, 1), (0, 2)): '<', ((0, 1), (1, 1)): '>', ((0, 2), (0, 3)): '>', ((0, 2), (1, 2)): '>', ((1, 0), (1, 1)): '>', ((1, 2), (2, 2)): '>', ((2, 0), (3, 0)): '<', ((2, 1), (2, 2)): '>', ((2, 2), (2, 3)): '<'}, 'solution': [[1, 3, 4, 2], [4, 2, 3, 1], [2, 4, 1, 3], [3, 1, 2, 4]], 'difficulty': {'board_size': 4, 'difficulty': 2}}

Example 3:
Question: Solve the following 7x7 Futoshiki puzzle:

_   _   _   _   7   _   _
                         
3   2   _   _   _   _   _
                        ∧
_   _   2   5   _   _   7
                         
_   _   6   _   1   5   _
                ∧        
4   _   _   2   6   1   _
                         
7   _   _   _   _   3   _
                         
_   5   _   _   _   2   4

Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.
Use < and > for horizontal constraints. Use ∧ and ∨ for vertical constraints.
Remember, in Futoshiki each row and column must contain each number from 1 to 7 exactly once.
Answer: 5   6   1   3   7   4   2
                         
3   2   4   6   5   7   1
                        ∧
1   3   2   5   4   6   7
                         
2   4   6   7   1   5   3
                ∧        
4   7   3   2   6   1   5
                         
7   1   5   4   2   3   6
                         
6   5   7   1   3   2   4
Metadata: {'puzzle': [[0, 0, 0, 0, 7, 0, 0], [3, 2, 0, 0, 0, 0, 0], [0, 0, 2, 5, 0, 0, 7], [0, 0, 6, 0, 1, 5, 0], [4, 0, 0, 2, 6, 1, 0], [7, 0, 0, 0, 0, 3, 0], [0, 5, 0, 0, 0, 2, 4]], 'constraints': {((1, 6), (2, 6)): '<', ((3, 4), (4, 4)): '<'}, 'solution': [[5, 6, 1, 3, 7, 4, 2], [3, 2, 4, 6, 5, 7, 1], [1, 3, 2, 5, 4, 6, 7], [2, 4, 6, 7, 1, 5, 3], [4, 7, 3, 2, 6, 1, 5], [7, 1, 5, 4, 2, 3, 6], [6, 5, 7, 1, 3, 2, 4]], 'difficulty': {'board_size': 7, 'difficulty': 0}}

````

### game_of_life
Generates Game of Life games with configurable parameters

Default configuration:
```python
grid_size_x = 10
grid_size_y = 10
filled_cells_weights = 0.1
filled_cells = 10
simulation_steps = 1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: What will this Game of Life board look like after 1 steps of simulation? Assume a Moore neighborhood and wrapping topology. Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

[[0,1,0,0,0,0,0,0,1,0],
 [1,0,0,0,0,0,0,0,1,1],
 [0,0,0,0,0,0,0,0,0,0],
 [0,0,1,1,0,0,0,0,0,0],
 [0,0,0,1,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0],
 [1,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,1],
 [0,0,0,0,0,0,0,0,0,0]].
Answer: [[1,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,0,0,1],[0,0,1,1,0,0,0,0,0,0],[0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
Metadata: {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells': 10, 'simulation_steps': 1, 'difficulty': {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells_weights': 0.1, 'simulation_steps': 1}}

Example 2:
Question: What will this Game of Life board look like after 1 steps of simulation? Assume a Moore neighborhood and wrapping topology. Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

[[0,0,0,0,1,0,0,0,1,0],
 [0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0],
 [0,1,0,0,0,0,0,0,0,1],
 [0,0,1,0,0,0,0,0,0,1],
 [0,0,0,0,0,0,0,0,0,1],
 [0,0,0,0,0,0,1,0,0,0],
 [0,0,0,0,0,0,0,0,0,0]].
Answer: [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
Metadata: {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells': 10, 'simulation_steps': 1, 'difficulty': {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells_weights': 0.1, 'simulation_steps': 1}}

Example 3:
Question: What will this Game of Life board look like after 1 steps of simulation? Assume a Moore neighborhood and wrapping topology. Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

[[0,1,0,1,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,1],
 [0,0,0,0,0,0,1,0,1,0],
 [0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0],
 [0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,1,0],
 [0,0,0,0,0,0,0,0,0,0],
 [0,1,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,1,0,0,0]].
Answer: [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,1,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0]]
Metadata: {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells': 10, 'simulation_steps': 1, 'difficulty': {'grid_size_x': 10, 'grid_size_y': 10, 'filled_cells_weights': 0.1, 'simulation_steps': 1}}

````

### game_of_life_halting
Generates Game of Life games with configurable parameters

    This is a variant of the Game of Life task, which rather than trying to test the algorithmic simulation, tests
    the ability of the model to do explanatory reasoning of the board. The idea is that a model with good
    explanatory reasoning will be able to see that a game will not halt without simulating it into the future.

    The task presents a GoL board, and the model is asked to predict if the board will halt (die, all cells zero)
    after n steps. Sometimes, the board will be made up of 'oscillators', isolated structures which never die.
    Othertimes, it is filled with non-oscillators, structures which will always die after a few steps. The model
    should deduce which case the presented board is.

Default configuration:
```python
grid_size_x = 12
grid_size_y = 12
difficulty = 1
num_oscillators = 5
max_simulation_steps = 20
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: This is a 'Game of Life' grid. We consider a game halted if there are no cells alive.
Will this game halt at or before 20 steps? Assume a Moore neighborhood and wrapping topology. If it will halt, reply 'True'. If it won't halt, reply 'False'.

Initial board:
[[0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]
Answer: False
Metadata: {'grid_size_x': 12, 'grid_size_y': 12, 'placed_patterns': [{'name': 'blinker', 'position': (5, 4)}], 'simulation_steps': 20, 'should_oscillate': True}

Example 2:
Question: This is a 'Game of Life' grid. We consider a game halted if there are no cells alive.
Will this game halt at or before 20 steps? Assume a Moore neighborhood and wrapping topology. If it will halt, reply 'True'. If it won't halt, reply 'False'.

Initial board:
[[0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0]
 [0 1 0 1 0 0 0 0 1 1 1 0]
 [0 0 1 0 1 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 1 1 0 0 0 1 0 1 0 0]
 [0 1 0 0 0 0 0 0 1 0 1 0]
 [0 0 0 0 1 0 0 0 1 0 0 0]
 [0 0 1 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]
Answer: False
Metadata: {'grid_size_x': 12, 'grid_size_y': 12, 'placed_patterns': [{'name': 'clock', 'position': (6, 7)}, {'name': 'toad', 'position': (7, 1)}, {'name': 'clock', 'position': (1, 1)}, {'name': 'blinker', 'position': (1, 8)}], 'simulation_steps': 20, 'should_oscillate': True}

Example 3:
Question: This is a 'Game of Life' grid. We consider a game halted if there are no cells alive.
Will this game halt at or before 20 steps? Assume a Moore neighborhood and wrapping topology. If it will halt, reply 'True'. If it won't halt, reply 'False'.

Initial board:
[[0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 0 0 0]
 [0 1 0 0 0 0 0 0 0 1 0 0]
 [0 0 1 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]
Answer: True
Metadata: {'grid_size_x': 12, 'grid_size_y': 12, 'placed_patterns': [{'name': 'non-oscillator', 'position': (5, 6)}, {'name': 'non-oscillator', 'position': (2, 1)}, {'name': 'non-oscillator', 'position': (8, 1)}], 'simulation_steps': 20, 'should_oscillate': False}

````

### gcd
Generates Greatest Common Divisor (GCD) tasks

Default configuration:
```python
min_numbers = 2
max_numbers = 2
min_value = 1
max_value = 1000
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 26, 760. Give only the GCD as your final answer.
Answer: 2
Metadata: {'numbers': [26, 760], 'result': 2, 'difficulty': {'num_terms': 2, 'max_value': 1000}}

Example 2:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 688, 716. Give only the GCD as your final answer.
Answer: 4
Metadata: {'numbers': [688, 716], 'result': 4, 'difficulty': {'num_terms': 2, 'max_value': 1000}}

Example 3:
Question: Find the Greatest Common Divisor (GCD) of these numbers: 297, 30. Give only the GCD as your final answer.
Answer: 3
Metadata: {'numbers': [297, 30], 'result': 3, 'difficulty': {'num_terms': 2, 'max_value': 1000}}

````

### graph_color
Generates graph coloring problems with configurable parameters

Default configuration:
```python
num_colors = 3
min_num_vertices = 10
max_num_vertices = 10
edge_probability = 0.1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties:

Vertices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Edges: [(0, 1), (0, 7), (0, 9), (1, 4), (2, 4), (3, 5), (3, 6), (6, 8), (7, 9)]
Possible colors: [1, 2, 3]

Return your solution as a JSON map of vertices to colors. (For example: {"0": 1, "1": 2, "2": 3}.)

Answer: None
Metadata: {'possible_answer': {0: 1, 1: 2, 2: 1, 3: 1, 4: 3, 5: 2, 6: 2, 7: 2, 8: 1, 9: 3}, 'puzzle': {'vertices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'edges': [(0, 1), (0, 7), (0, 9), (1, 4), (2, 4), (3, 5), (3, 6), (6, 8), (7, 9)], 'num_colors': 3, 'color_options': [1, 2, 3]}, 'difficulty': {'num_vertices': 10, 'num_colors': 3}}

Example 2:
Question: Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties:

Vertices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Edges: [(0, 6), (1, 8), (3, 6), (3, 9), (4, 7), (5, 9)]
Possible colors: [1, 2, 3]

Return your solution as a JSON map of vertices to colors. (For example: {"0": 1, "1": 2, "2": 3}.)

Answer: None
Metadata: {'possible_answer': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}, 'puzzle': {'vertices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'edges': [(0, 6), (1, 8), (3, 6), (3, 9), (4, 7), (5, 9)], 'num_colors': 3, 'color_options': [1, 2, 3]}, 'difficulty': {'num_vertices': 10, 'num_colors': 3}}

Example 3:
Question: Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties:

Vertices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Edges: [(3, 7), (4, 8)]
Possible colors: [1, 2, 3]

Return your solution as a JSON map of vertices to colors. (For example: {"0": 1, "1": 2, "2": 3}.)

Answer: None
Metadata: {'possible_answer': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 1}, 'puzzle': {'vertices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'edges': [(3, 7), (4, 8)], 'num_colors': 3, 'color_options': [1, 2, 3]}, 'difficulty': {'num_vertices': 10, 'num_colors': 3}}

````

### group_anagrams
Generates Group Anagrams exercises with configurable difficulty

Default configuration:
```python
min_anagram_groups = 2
max_anagram_groups = 10
min_words_per_group = 2
max_words_per_group = 5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

The output is a list of lists of strings, where each outer list contains a group of anagrams, e.g. [["eat", "tea"], ["tan", "nat"]].

Group the following list of words into anagrams:
["ditas", "adits", "staid", "plastinoid", "palinodist", "cytomere", "merocyte"]

Answer: [["adits", "ditas", "staid"], ["cytomere", "merocyte"], ["palinodist", "plastinoid"]]
Metadata: {'words': ['ditas', 'adits', 'staid', 'plastinoid', 'palinodist', 'cytomere', 'merocyte'], 'solution': [['adits', 'ditas', 'staid'], ['cytomere', 'merocyte'], ['palinodist', 'plastinoid']], 'difficulty': {'anagram_groups': 3}}

Example 2:
Question: An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

The output is a list of lists of strings, where each outer list contains a group of anagrams, e.g. [["eat", "tea"], ["tan", "nat"]].

Group the following list of words into anagrams:
["escrod", "decors", "scored", "semitaur", "muriates"]

Answer: [["decors", "escrod", "scored"], ["muriates", "semitaur"]]
Metadata: {'words': ['escrod', 'decors', 'scored', 'semitaur', 'muriates'], 'solution': [['decors', 'escrod', 'scored'], ['muriates', 'semitaur']], 'difficulty': {'anagram_groups': 2}}

Example 3:
Question: An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

The output is a list of lists of strings, where each outer list contains a group of anagrams, e.g. [["eat", "tea"], ["tan", "nat"]].

Group the following list of words into anagrams:
["granitite", "iterating", "helium", "humlie", "nizam", "nazim", "striplings", "slipstring", "rearrest", "arrester", "bf", "fb", "tadpolism", "diplomats", "cunan", "canun"]

Answer: [["arrester", "rearrest"], ["bf", "fb"], ["canun", "cunan"], ["diplomats", "tadpolism"], ["granitite", "iterating"], ["helium", "humlie"], ["nazim", "nizam"], ["slipstring", "striplings"]]
Metadata: {'words': ['granitite', 'iterating', 'helium', 'humlie', 'nizam', 'nazim', 'striplings', 'slipstring', 'rearrest', 'arrester', 'bf', 'fb', 'tadpolism', 'diplomats', 'cunan', 'canun'], 'solution': [['arrester', 'rearrest'], ['bf', 'fb'], ['canun', 'cunan'], ['diplomats', 'tadpolism'], ['granitite', 'iterating'], ['helium', 'humlie'], ['nazim', 'nizam'], ['slipstring', 'striplings']], 'difficulty': {'anagram_groups': 8}}

````

### gsm_symbolic
Default configuration:
```python
difficulty = 1.0
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: There are 12 students playing basketball and twice that number playing volleyball. There are 17 boys and 17 girls playing table tennis. If each student only participates in one group, how many students are there in total? Give the result as your final answer. Do not include units.
Answer: 70
Metadata: {'difficulty': 1.0, 'answer_value': 70, 'answer_cot': 'There are 12 x 2 = 24 students playing volleyball.\nThere are 17 + 17 = 34 students playing table tennis.\nIn total there are 12 + 24 + 34 = 70 students.\n#### 70', 'variables': {'tennis_players': 12, 'volleyball_players': 24, 'soccer_boys': 17, 'soccer_girls': 17, 'total_soccer': 34, 'total_students': 70, 'sports': ['basketball', 'volleyball', 'table tennis']}}

Example 2:
Question: In Ms. Johnson's class of 100 students, 80% of the class are volleyball players. Out of the remaining class, 65% of the students are choir members or part of robotics club members. These 3 groups of students will need to leave early today to travel to an away performance. How many students are leaving early? Give the result as your final answer. Do not include units.
Answer: 93
Metadata: {'difficulty': 1.0, 'answer_value': 93, 'answer_cot': "80% of the 100 student class are volleyball players so that's 0.8*100 = 80 students\nThere are 100 students and 80 are volleyball players so that leaves 100-80 = 20 students\n65% of the remaining 20 students are part of robotics club members or choir members so that's 0.65*20 = 13 students\n80 students are volleyball players and 13 are part of robotics club members/choir members so 80+13 = 93 students will be leaving early\n#### 93", 'variables': {'teacher': 'Ms. Johnson', 'total_students': 100, 'percent_group1': 80, 'percent_group23': 65, 'group1': 'volleyball players', 'group2': 'choir members', 'group3': 'robotics club members', 'event': 'performance', 'group1_count': 80, 'group23_count': 13}}

Example 3:
Question: Olivia is trying to decide whether to do her business accounting herself or hire an accountant. If she does it herself, she'll be able to do 7 fewer hours of consulting work, losing €57/hour in missed income. The accountant charges €57. How much more money will she have if she hires the accountant? Give the result as your final answer. Do not include units.
Answer: 342
Metadata: {'difficulty': 1.0, 'answer_value': 342, 'answer_cot': "First find the total lost revenue if Olivia does her business accounting herself: €57/hour * 7 hours = €399\nThen subtract the accountant's charge to find how much money Olivia saves: €399 - €57 = €342\n#### 342", 'variables': {'name': 'Olivia', 'task': 'her business accounting', 'profession': 'accountant', 'hours': 7, 'work_type': 'consulting', 'hourly_rate': 57, 'fee': 57, 'currency': '€', 'lost_income': 399}}

````

### intermediate_integration
Generates intermediate integration problem - either
    by substitution or by parts

Default configuration:
```python
problem_types = ('linear', 'radical', 'log_inverse_trig', 'trigonometric', 'polynomial_exp_trig', 'exponential', 'cyclic', 'repeated_parts')
problem_type_weights = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
seed = 42
size = 500
linear_lower_bound = 1
linear_upper_bound = 10
min_linear_degree = 2
max_linear_degree = 4
outer_constant_min = 1
outer_constant_max = 3
min_poly_degree = 1
max_poly_degree = 3
symbols = x
operators = ('+', '-')
```

Example tasks:
````
Example 1:
Question: Find the indefinite integral: ∫ (16*x + 4)*exp(4*x**2 + 2*x + 10) dx
When performing calculations, please follow these guidelines:
Use same variable symbols as given in the question
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.
3. Use `exp(x)` or `E**(x)` for the exponential function (i.e. use capital E for Euler's number).

Answer: 2*exp(4*x**2 + 2*x + 10) + C
Metadata: {'integrand': '(16*x + 4)*exp(4*x**2 + 2*x + 10)', 'problem_type': 'exponential', 'variable': 'x', 'expected_answer_expression': 2*exp(4*x**2 + 2*x + 10), 'difficulty': {'problem_type_weights': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]}}

Example 2:
Question: Evaluate the indefinite integral: ∫ -3*(8*x + 6)**3 dx
When performing calculations, please follow these guidelines:
Use same variable symbols as given in the question
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.
3. Use `exp(x)` or `E**(x)` for the exponential function (i.e. use capital E for Euler's number).

Answer: -384*x**4 - 1152*x**3 - 1296*x**2 - 648*x + C
Metadata: {'integrand': '-3*(8*x + 6)**3', 'problem_type': 'linear', 'variable': 'x', 'expected_answer_expression': -384*x**4 - 1152*x**3 - 1296*x**2 - 648*x, 'difficulty': {'problem_type_weights': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]}}

Example 3:
Question: Find the indefinite integral: ∫ -4*sin(4*x + 1)**2*cos(4*x + 1) dx
When performing calculations, please follow these guidelines:
Use same variable symbols as given in the question
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.
3. Use `exp(x)` or `E**(x)` for the exponential function (i.e. use capital E for Euler's number).

Answer: -sin(4*x + 1)**3/3 + C
Metadata: {'integrand': '-4*sin(4*x + 1)**2*cos(4*x + 1)', 'problem_type': 'trigonometric', 'variable': 'x', 'expected_answer_expression': -sin(4*x + 1)**3/3, 'difficulty': {'problem_type_weights': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]}}

````

### isomorphic_strings
Generates Isomorphic Strings exercises with configurable difficulty

Default configuration:
```python
min_string_length = 2
max_string_length = 10
p_solvable = 0.5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Return True if the following two strings are isomorphic, or False otherwise:
cj oe

Answer: True
Metadata: {'words': ['cj', 'oe'], 'solution': True, 'solvable': True, 'difficulty': {'string_length': 3}}

Example 2:
Question: Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Return True if the following two strings are isomorphic, or False otherwise:
t e

Answer: True
Metadata: {'words': ['t', 'e'], 'solution': True, 'solvable': True, 'difficulty': {'string_length': 2}}

Example 3:
Question: Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Return True if the following two strings are isomorphic, or False otherwise:
sdjxdqxq lcdqmtqt

Answer: False
Metadata: {'words': ['sdjxdqxq', 'lcdqmtqt'], 'solution': False, 'solvable': False, 'difficulty': {'string_length': 8}}

````

### jugs
Generates water jug puzzles inspired by [this scene from _Die Hard 3_](https://www.youtube.com/watch?v=6cAbgAaEOVE), with configurable parameters

Default configuration:
```python
num_jugs = 3
difficulty = 10
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You are a police officer. A maniac has planted a bomb next to a public fountain.

To defuse the bomb, you must solve a puzzle. The puzzle is solved when you fill any of the available jugs with the target amount of water.

You have three move types: 'fill', 'empty' and 'pour'.

To fill Jug A, you 'fill A'.
To empty Jug B, you 'empty B'.
To pour the contents of Jug A into Jug B, you 'pour A->B'.
All jugs are empty to begin with.

The empty jugs hold this many litres of water: A:13, B:13, C:4
And your target is: 10 litres.

How do you defuse the bomb?

Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.

Answer: ["fill A", "pour A->C", "fill B", "empty C", "pour A->C", "empty C", "pour A->C", "empty C", "pour A->C", "pour B->C"]
Metadata: {'puzzle': {'jug_capacities': [13, 13, 4], 'target': 10, 'min_moves': 10}, 'difficulty': {'num_jugs': 3, 'difficulty': 10}}

Example 2:
Question: You are a police officer. A maniac has planted a bomb next to a public fountain.

To defuse the bomb, you must solve a puzzle. The puzzle is solved when you fill any of the available jugs with the target amount of water.

You have three move types: 'fill', 'empty' and 'pour'.

To fill Jug A, you 'fill A'.
To empty Jug B, you 'empty B'.
To pour the contents of Jug A into Jug B, you 'pour A->B'.
All jugs are empty to begin with.

The empty jugs hold this many litres of water: A:7, B:10, C:10
And your target is: 5 litres.

How do you defuse the bomb?

Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.

Answer: ["fill A", "pour A->B", "fill A", "pour A->B", "pour A->C", "fill A", "pour A->C", "empty B", "pour A->B", "fill A", "pour A->B", "fill A", "pour A->B"]
Metadata: {'puzzle': {'jug_capacities': [7, 10, 10], 'target': 5, 'min_moves': 13}, 'difficulty': {'num_jugs': 3, 'difficulty': 10}}

Example 3:
Question: You are a police officer. A maniac has planted a bomb next to a public fountain.

To defuse the bomb, you must solve a puzzle. The puzzle is solved when you fill any of the available jugs with the target amount of water.

You have three move types: 'fill', 'empty' and 'pour'.

To fill Jug A, you 'fill A'.
To empty Jug B, you 'empty B'.
To pour the contents of Jug A into Jug B, you 'pour A->B'.
All jugs are empty to begin with.

The empty jugs hold this many litres of water: A:7, B:10, C:7
And your target is: 2 litres.

How do you defuse the bomb?

Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.

Answer: ["fill B", "pour B->A", "empty A", "pour B->A", "fill B", "pour B->A", "empty A", "pour B->A", "fill B", "pour B->A", "pour B->C"]
Metadata: {'puzzle': {'jug_capacities': [7, 10, 7], 'target': 2, 'min_moves': 11}, 'difficulty': {'num_jugs': 3, 'difficulty': 10}}

````

### knight_swap
Generates Knight Swap puzzles with configurable parameters.

Default configuration:
```python
min_nodes = 6
max_nodes = 9
min_pieces = 2
max_pieces = 2
min_steps = 4
max_steps = 20
max_attempts = 100
seed = 42
size = 5
impossible_ratio = 0.2
```

Example tasks:
````
Example 1:
Question: Knight Swap Challenge:

```
    A   B   C   D
   ----------------
3 |   | . |   | . |
   ----------------
2 | B | w |   |   |
   ----------------
1 |   |   | B | w |
   ----------------
```

Legend:
- 'w' = White Knight
- 'B' = Black Knight
- Empty squares are marked with '.'

Objective:
Swap the positions of all white knights with all black knights through valid moves.

Rules:
1. Knights move in L-shape (2 squares + 1 square perpendicular)
2. Knights can only move to empty squares
3. w moves first, then players alternate
4. All knights must reach their target positions (white ↔ black)

Question:
Is it possible to swap all knights' positions? If yes, list the moves.

Answer Format:
- For impossible puzzles: "No"
- For possible puzzles: List moves as ["color,from,to", ...]
  Example: ["w,A1,B3"] means white knight moves A1→B3

Answer: No
Metadata: {'board': {'C1': ['A2', 'B3', 'D3'], 'A2': ['C1'], 'B3': ['C1'], 'D1': ['B2'], 'B2': ['D1', 'D3'], 'D3': ['B2', 'C1']}, 'pieces': {'C1': 'B', 'A2': 'B', 'B3': None, 'D1': 'w', 'B2': 'w', 'D3': None}, 'start_turn': 'w', 'solution': None, 'is_possible': False, 'num_steps': 0, 'board_states': None}

Example 2:
Question: Knight Swap Challenge:

```
    A   B   C   D
   ----------------
3 |   | w | . |   |
   ----------------
2 | w |   |   | B |
   ----------------
1 |   |   | . | B |
   ----------------
```

Legend:
- 'w' = White Knight
- 'B' = Black Knight
- Empty squares are marked with '.'

Objective:
Swap the positions of all white knights with all black knights through valid moves.

Rules:
1. Knights move in L-shape (2 squares + 1 square perpendicular)
2. Knights can only move to empty squares
3. w moves first, then players alternate
4. All knights must reach their target positions (white ↔ black)

Question:
Is it possible to swap all knights' positions? If yes, list the moves.

Answer Format:
- For impossible puzzles: "No"
- For possible puzzles: List moves as ["color,from,to", ...]
  Example: ["w,A1,B3"] means white knight moves A1→B3

Answer: No
Metadata: {'board': {'B3': ['C1'], 'D1': ['C3'], 'C3': ['A2', 'D1'], 'C1': ['A2', 'B3'], 'D2': [], 'A2': ['C1', 'C3']}, 'pieces': {'B3': 'w', 'D1': 'B', 'C3': None, 'C1': None, 'D2': 'B', 'A2': 'w'}, 'start_turn': 'w', 'solution': None, 'is_possible': False, 'num_steps': 0, 'board_states': None}

Example 3:
Question: Knight Swap Challenge:

```
    A   B   C
   ------------
3 | . |   | B |
   ------------
2 | w |   | . |
   ------------
1 |   | w | B |
   ------------
```

Legend:
- 'w' = White Knight
- 'B' = Black Knight
- Empty squares are marked with '.'

Objective:
Swap the positions of all white knights with all black knights through valid moves.

Rules:
1. Knights move in L-shape (2 squares + 1 square perpendicular)
2. Knights can only move to empty squares
3. w moves first, then players alternate
4. All knights must reach their target positions (white ↔ black)

Question:
Is it possible to swap all knights' positions? If yes, list the moves.

Answer Format:
- For impossible puzzles: "No"
- For possible puzzles: List moves as ["color,from,to", ...]
  Example: ["w,A1,B3"] means white knight moves A1→B3

Answer: No
Metadata: {'board': {'B1': ['A3'], 'A3': ['B1', 'C2'], 'A2': ['C1', 'C3'], 'C3': ['A2'], 'C1': ['A2'], 'C2': ['A3']}, 'pieces': {'B1': 'w', 'A3': None, 'A2': 'w', 'C3': 'B', 'C1': 'B', 'C2': None}, 'start_turn': 'w', 'solution': None, 'is_possible': False, 'num_steps': 0, 'board_states': None}

````

### knights_knaves
Generates random knights and knaves problems.

    This implementation is adapted from the Knights and Knaves problem generator in:
    https://github.com/AlphaPav/mem-kk-logic

    As described in the paper:
    @article{xie2024memorization,
    title={On Memorization of Large Language Models in Logical Reasoning},
    author={Chulin Xie and Yangsibo Huang and Chiyuan Zhang and Da Yu and Xinyun Chen and Bill Yuchen Lin and Bo Li and Badih Ghazi and Ravi Kumar},
    year={2024},
    eprint={2410.23123},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2410.23123},
    }

Default configuration:
```python
n_people = 2
depth_constraint = 2
width_constraint = 2
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: A very special island is inhabited only by sages and fools. Sages always tell the truth, and fools always lie. You meet 2 inhabitants: Zoey, and Riley. Zoey commented, "Riley is a fool". In Riley's words: "Zoey is a sage or Riley is a sage". So who is a sage and who is a fool? (Format your answer like: "Zoey is a sage/fool, and Riley is a sage/fool")
Answer: Zoey is a fool, and Riley is a sage.
Metadata: {'statements': (('lying', 1), ('or', ('telling-truth', 0), ('telling-truth', 1))), 'solution': (False, True), 'names': ['Zoey', 'Riley'], 'knight_knave_terms': {'knight': 'sage', 'knave': 'fool', 'a_knight': 'a sage', 'a_knave': 'a fool', 'Knight': 'Sage', 'Knave': 'Fool'}}

Example 2:
Question: A very special island is inhabited only by pioneers and laggards. Pioneers always tell the truth, and laggards always lie. You meet 2 inhabitants: Riley, and Olivia. "if Riley is a pioneer then Olivia is a laggard" - Riley. Olivia stated, "Olivia is a pioneer and Riley is a laggard". So who is a pioneer and who is a laggard? (Format your answer like: "Riley is a pioneer/laggard, and Olivia is a pioneer/laggard")
Answer: Riley is a pioneer, and Olivia is a laggard.
Metadata: {'statements': (('->', ('telling-truth', 0), ('lying', 1)), ('and', ('telling-truth', 1), ('lying', 0))), 'solution': (True, False), 'names': ['Riley', 'Olivia'], 'knight_knave_terms': {'knight': 'pioneer', 'knave': 'laggard', 'a_knight': 'a pioneer', 'a_knave': 'a laggard', 'Knight': 'Pioneer', 'Knave': 'Laggard'}}

Example 3:
Question: A very special island is inhabited only by saints and sinners. Saints always tell the truth, and sinners always lie. You meet 2 inhabitants: Samuel, and Jacob. Samuel expressed that if Samuel is a saint then Jacob is a sinner. Jacob was heard saying, "if Samuel is a saint then Samuel is a sinner". So who is a saint and who is a sinner? (Format your answer like: "Samuel is a saint/sinner, and Jacob is a saint/sinner")
Answer: Samuel is a saint, and Jacob is a sinner.
Metadata: {'statements': (('->', ('telling-truth', 0), ('lying', 1)), ('->', ('telling-truth', 0), ('lying', 0))), 'solution': (True, False), 'names': ['Samuel', 'Jacob'], 'knight_knave_terms': {'knight': 'saint', 'knave': 'sinner', 'a_knight': 'a saint', 'a_knave': 'a sinner', 'Knight': 'Saint', 'Knave': 'Sinner'}}

````

### largest_island
Generates Largest Island exercises with configurable difficulty

Default configuration:
```python
min_rows = 5
max_rows = 10
min_cols = 5
max_cols = 10
min_num_islands = 0
max_num_islands = 5
min_island_size = 0
max_island_size = 10
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: You are given the following 10 x 5 binary matrix grid:
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Answer: 0
Metadata: {'grid': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'solution': 0, 'difficulty': {'rows': 10, 'cols': 5, 'num_islands': 0}}

Example 2:
Question: You are given the following 5 x 7 binary matrix grid:
0 0 1 1 1 0 1
1 0 0 1 1 0 1
1 0 0 0 0 1 1
1 0 0 1 1 1 0
1 0 0 1 1 1 0

An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Answer: 10
Metadata: {'grid': [[0, 0, 1, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 0, 0, 1, 1], [1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0]], 'solution': 10, 'difficulty': {'rows': 5, 'cols': 7, 'num_islands': 5}}

Example 3:
Question: You are given the following 8 x 9 binary matrix grid:
1 0 0 0 0 0 0 0 0
1 1 1 0 0 0 0 0 1
1 1 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Answer: 6
Metadata: {'grid': [[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 'solution': 6, 'difficulty': {'rows': 8, 'cols': 9, 'num_islands': 4}}

````

### lcm
Generates Least Common Multiple (LCM) tasks

Default configuration:
```python
min_numbers = 2
max_numbers = 2
min_value = 1
max_value = 100
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the Least Common Multiple (LCM) of these numbers: 95, 14
Answer: 1330
Metadata: {'numbers': [95, 14], 'result': 1330, 'difficulty': {'numbers': 2, 'value': (1, 100)}}

Example 2:
Question: Find the Least Common Multiple (LCM) of these numbers: 60, 48
Answer: 240
Metadata: {'numbers': [60, 48], 'result': 240, 'difficulty': {'numbers': 2, 'value': (1, 100)}}

Example 3:
Question: Find the Least Common Multiple (LCM) of these numbers: 38, 4
Answer: 76
Metadata: {'numbers': [38, 4], 'result': 76, 'difficulty': {'numbers': 2, 'value': (1, 100)}}

````

### leg_counting
Generates leg counting arithmetic tasks

Default configuration:
```python
min_animals = 3
max_animals = 10
min_instances = 1
max_instances = 15
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to count how many legs there are in total when given a list of animals.

Now, how many legs are there in total if you have 3 sea slugs, 12 deers, 2 giraffes, 11 elephants?

Answer: 100
Metadata: {'difficulty': {'num_animals': 4}, 'animals': {'sea slug': 3, 'deer': 12, 'giraffe': 2, 'elephant': 11}, 'total_legs': 100}

Example 2:
Question: Your task is to count how many legs there are in total when given a list of animals.

Now, how many legs are there in total if you have 6 sheeps, 11 dogs, 12 praying mantiss?

Answer: 140
Metadata: {'difficulty': {'num_animals': 3}, 'animals': {'sheep': 6, 'dog': 11, 'praying mantis': 12}, 'total_legs': 140}

Example 3:
Question: Your task is to count how many legs there are in total when given a list of animals.

Now, how many legs are there in total if you have 2 crabs, 10 lobsters, 1 human, 2 cows, 3 bees, 13 elephants, 9 dogs, 12 snakes, 5 shrimps?

Answer: 286
Metadata: {'difficulty': {'num_animals': 9}, 'animals': {'crab': 2, 'lobster': 10, 'human': 1, 'cow': 2, 'bee': 3, 'elephant': 13, 'dog': 9, 'snake': 12, 'shrimp': 5}, 'total_legs': 286}

````

### letter_counting
Generates letter counting tasks from text spans

Default configuration:
```python
min_words = 5
max_words = 15
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: How many times does the letter "a" appear in the text: "bed and enters his mechanical dresser Two minutes later the machine deposited him all dressed"?
Answer: 6
Metadata: {'span_length': 15, 'target_letter': 'a', 'span': ['bed', 'and', 'enters', 'his', 'mechanical', 'dresser', 'Two', 'minutes', 'later', 'the', 'machine', 'deposited', 'him', 'all', 'dressed'], 'difficulty': {'words': 15}}

Example 2:
Question: How many times does the letter "w" appear in the text: "it into a watering place"?
Answer: 1
Metadata: {'span_length': 5, 'target_letter': 'w', 'span': ['it', 'into', 'a', 'watering', 'place'], 'difficulty': {'words': 5}}

Example 3:
Question: How many times does the letter "t" appear in the text: "readable form accessible by the widest array of equipment including outdated"?
Answer: 5
Metadata: {'span_length': 11, 'target_letter': 't', 'span': ['readable', 'form', 'accessible', 'by', 'the', 'widest', 'array', 'of', 'equipment', 'including', 'outdated'], 'difficulty': {'words': 11}}

````

### letter_jumble
Generates word letter jumbling tasks

Default configuration:
```python
min_word_len = 1
max_word_len = 64
min_words = 3
max_words = 20
min_corruption_level = 0.1
max_corruption_level = 0.9
consecutive_words = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to unsramble words in a sentence.

For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the words.

The order of the words in the sentence is preserved. Moreover, the style of the sentence is preserved (i.e. punctuation, capitalization, new lines, etc.).

Your output should be a sentence with the words unscrambled.

Now, unscramble these words: ew hsall eb ebla ot puodrce

Answer: we shall be able to produce
Metadata: {'num_words': 6, 'corruption_level': 0.12000860417813355, 'scrambled_words': ['ew', 'hsall', 'eb', 'ebla', 'ot', 'puodrce'], 'original_words': ['we', 'shall', 'be', 'able', 'to', 'produce'], 'difficulty': {'word_len': (1, 64), 'words': 6, 'corruption_level': 0.12000860417813355}}

Example 2:
Question: Your task is to unsramble words in a sentence.

For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the words.

The order of the words in the sentence is preserved. Moreover, the style of the sentence is preserved (i.e. punctuation, capitalization, new lines, etc.).

Your output should be a sentence with the words unscrambled.

Now, unscramble these words: ni oiurnalmsj Well Cahs

Answer: in journalism Well Cash
Metadata: {'num_words': 4, 'corruption_level': 0.3288673442377109, 'scrambled_words': ['ni', 'oiurnalmsj', 'Well', 'Cahs'], 'original_words': ['in', 'journalism', 'Well', 'Cash'], 'difficulty': {'word_len': (1, 64), 'words': 4, 'corruption_level': 0.3288673442377109}}

Example 3:
Question: Your task is to unsramble words in a sentence.

For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the words.

The order of the words in the sentence is preserved. Moreover, the style of the sentence is preserved (i.e. punctuation, capitalization, new lines, etc.).

Your output should be a sentence with the words unscrambled.

Now, unscramble these words: dear rchAdbali keep no nSice yrstyedae atnhks ot oyu rheet si a gain fo sucrbbisesr rM

Answer: dear Archibald keep on Since yesterday thanks to you there is a gain of subscribers Mr
Metadata: {'num_words': 16, 'corruption_level': 0.516016391169858, 'scrambled_words': ['dear', 'rchAdbali', 'keep', 'no', 'nSice', 'yrstyedae', 'atnhks', 'ot', 'oyu', 'rheet', 'si', 'a', 'gain', 'fo', 'sucrbbisesr', 'rM'], 'original_words': ['dear', 'Archibald', 'keep', 'on', 'Since', 'yesterday', 'thanks', 'to', 'you', 'there', 'is', 'a', 'gain', 'of', 'subscribers', 'Mr'], 'difficulty': {'word_len': (1, 64), 'words': 16, 'corruption_level': 0.516016391169858}}

````

### list_functions
Default configuration:
```python
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [4, 95, 36, 32]
Output 1: [4, 32, 36, 95]
Input 2: [18, 95, 14, 87, 95, 70]
Output 2: [14, 18, 70, 87, 95, 95]
Input 3: [76, 55, 5, 4]
Output 3: [4, 5, 55, 76]
Input 4: [28, 30, 65, 78]
Output 4: [28, 30, 65, 78]


Input: [72, 26, 92]
Output:

Answer: [26, 72, 92]

Example 2:
Question: You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [37, 90, 98]
Output 1: [37, 90, 98]
Input 2: [60, 48, 86, 90, 13]
Output 2: [60, 48, 86, 90, 13]
Input 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Output 3: [77, 64, 78, 3, 66, 56, 74, 48, 80, 71]
Input 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]
Output 4: [51, 23, 8, 14, 16, 49, 20, 13, 21]


Input: [17, 99, 50, 77, 65, 35, 74, 24, 49, 9]
Output:

Answer: [17, 99, 50, 77, 65, 35, 74, 24, 49, 9]

Example 3:
Question: You are an expert at inductive reasoning. Generate an output corresponding to the given input.
The output is generated by applying the same rule that maps input to output for the examples provided. Your answer should be a list of element/elements
Examples:
Input 1: [4, 29, 49, 15, 90, 23, 38, 5, 67, 5, 70]
Output 1: [2]
Input 2: [37, 66, 21, 15, 44, 46, 80, 10]
Output 2: [0]
Input 3: [13, 45, 5, 5, 5, 50, 5]
Output 3: [4]
Input 4: [88, 6, 87]
Output 4: [0]


Input: [59, 5, 81, 5, 20, 5, 61, 76, 48, 70, 5, 30]
Output:

Answer: [4]

````

### mahjong_puzzle
Generates Mahjong Puzzle exercises with configurable difficulty

Default configuration:
```python
min_num_rounds = 10
max_num_rounds = 50
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: There are several letter cards, and the game rules are as follows:
1. Initially, there are 13 cards.
2. Each time, a new card is added, and a result is determined. Then, one card is removed.
3. When there are two identical cards in hand, and the newly added card is the same as these two cards, the result is determined as "Peng".
4. If there are two cards in hand such that the new card can form a consecutive letter sequence with these two cards, the result is determined as "Chi". For example: ABC, BCD, CDE, etc.
5. If the new card does not meet the conditions of 3 and 4, the result is determined as "Pass".
6. "Peng" takes precedence over "Chi".
7. The card that is removed does not affect the result determination of the current round.

Your output should be one of the following: "Peng", "Chi", or "Pass" (without quotes).

Now, given the initial cards OHBVRPOIGIFLB, what is the result at the end of performing the following rounds of operations:
Round 1: Add a B card and remove a V card.
Round 2: Add an O card and remove a F card.
Round 3: Add an O card and remove a R card.
Round 4: Add an U card and remove an O card.
Round 5: Add a N card and remove an O card.
Round 6: Add a Q card and remove a G card.
Round 7: Add a B card and remove a N card.
Round 8: Add a Q card and remove an I card.
Round 9: Add a B card and remove a Q card.
Round 10: Add a G card and remove a B card.
Round 11: Add a F card and remove a B card.
Round 12: Add an I card and remove a F card.

Answer: Chi
Metadata: {'rounds': [{'add': 'B', 'remove': 'V', 'cards': 'OHBRPOIGIFLBB', 'result': 'Peng'}, {'add': 'O', 'remove': 'F', 'cards': 'OHBRPOIGILBBO', 'result': 'Peng'}, {'add': 'O', 'remove': 'R', 'cards': 'OHBPOIGILBBOO', 'result': 'Peng'}, {'add': 'U', 'remove': 'O', 'cards': 'HBPOIGILBBOOU', 'result': 'Pass'}, {'add': 'N', 'remove': 'O', 'cards': 'HBPIGILBBOOUN', 'result': 'Pass'}, {'add': 'Q', 'remove': 'G', 'cards': 'HBPIILBBOOUNQ', 'result': 'Chi'}, {'add': 'B', 'remove': 'N', 'cards': 'HBPIILBBOOUQB', 'result': 'Peng'}, {'add': 'Q', 'remove': 'I', 'cards': 'HBPILBBOOUQBQ', 'result': 'Chi'}, {'add': 'B', 'remove': 'Q', 'cards': 'HBPILBBOOUBQB', 'result': 'Peng'}, {'add': 'G', 'remove': 'B', 'cards': 'HPILBBOOUBQBG', 'result': 'Chi'}, {'add': 'F', 'remove': 'B', 'cards': 'HPILBOOUBQBGF', 'result': 'Chi'}, {'add': 'I', 'remove': 'F', 'cards': 'HPILBOOUBQBGI', 'result': 'Chi'}], 'solution': 'Chi', 'difficulty': {'num_rounds': 12}}

Example 2:
Question: There are several letter cards, and the game rules are as follows:
1. Initially, there are 13 cards.
2. Each time, a new card is added, and a result is determined. Then, one card is removed.
3. When there are two identical cards in hand, and the newly added card is the same as these two cards, the result is determined as "Peng".
4. If there are two cards in hand such that the new card can form a consecutive letter sequence with these two cards, the result is determined as "Chi". For example: ABC, BCD, CDE, etc.
5. If the new card does not meet the conditions of 3 and 4, the result is determined as "Pass".
6. "Peng" takes precedence over "Chi".
7. The card that is removed does not affect the result determination of the current round.

Your output should be one of the following: "Peng", "Chi", or "Pass" (without quotes).

Now, given the initial cards CSSWJDXQGUMFP, what is the result at the end of performing the following rounds of operations:
Round 1: Add a N card and remove an U card.
Round 2: Add a V card and remove a X card.
Round 3: Add a S card and remove a S card.
Round 4: Add a S card and remove a W card.
Round 5: Add a S card and remove a P card.
Round 6: Add an E card and remove a S card.
Round 7: Add a R card and remove a V card.
Round 8: Add a P card and remove a D card.
Round 9: Add an E card and remove a C card.
Round 10: Add a B card and remove a G card.
Round 11: Add an E card and remove a N card.

Answer: Peng
Metadata: {'rounds': [{'add': 'N', 'remove': 'U', 'cards': 'CSSWJDXQGMFPN', 'result': 'Pass'}, {'add': 'V', 'remove': 'X', 'cards': 'CSSWJDQGMFPNV', 'result': 'Chi'}, {'add': 'S', 'remove': 'S', 'cards': 'CSWJDQGMFPNVS', 'result': 'Peng'}, {'add': 'S', 'remove': 'W', 'cards': 'CSJDQGMFPNVSS', 'result': 'Peng'}, {'add': 'S', 'remove': 'P', 'cards': 'CSJDQGMFNVSSS', 'result': 'Peng'}, {'add': 'E', 'remove': 'S', 'cards': 'CJDQGMFNVSSSE', 'result': 'Chi'}, {'add': 'R', 'remove': 'V', 'cards': 'CJDQGMFNSSSER', 'result': 'Chi'}, {'add': 'P', 'remove': 'D', 'cards': 'CJQGMFNSSSERP', 'result': 'Chi'}, {'add': 'E', 'remove': 'C', 'cards': 'JQGMFNSSSERPE', 'result': 'Chi'}, {'add': 'B', 'remove': 'G', 'cards': 'JQMFNSSSERPEB', 'result': 'Pass'}, {'add': 'E', 'remove': 'N', 'cards': 'JQMFSSSERPEBE', 'result': 'Peng'}], 'solution': 'Peng', 'difficulty': {'num_rounds': 11}}

Example 3:
Question: There are several letter cards, and the game rules are as follows:
1. Initially, there are 13 cards.
2. Each time, a new card is added, and a result is determined. Then, one card is removed.
3. When there are two identical cards in hand, and the newly added card is the same as these two cards, the result is determined as "Peng".
4. If there are two cards in hand such that the new card can form a consecutive letter sequence with these two cards, the result is determined as "Chi". For example: ABC, BCD, CDE, etc.
5. If the new card does not meet the conditions of 3 and 4, the result is determined as "Pass".
6. "Peng" takes precedence over "Chi".
7. The card that is removed does not affect the result determination of the current round.

Your output should be one of the following: "Peng", "Chi", or "Pass" (without quotes).

Now, given the initial cards AHISHLYOSBWVK, what is the result at the end of performing the following rounds of operations:
Round 1: Add a H card and remove a K card.
Round 2: Add a W card and remove a H card.
Round 3: Add an U card and remove an O card.
Round 4: Add a M card and remove an U card.
Round 5: Add a V card and remove a Y card.
Round 6: Add a C card and remove a S card.
Round 7: Add a K card and remove a H card.
Round 8: Add a V card and remove a M card.
Round 9: Add a W card and remove a W card.
Round 10: Add a W card and remove an A card.

Answer: Peng
Metadata: {'rounds': [{'add': 'H', 'remove': 'K', 'cards': 'AHISHLYOSBWVH', 'result': 'Peng'}, {'add': 'W', 'remove': 'H', 'cards': 'AISHLYOSBWVHW', 'result': 'Pass'}, {'add': 'U', 'remove': 'O', 'cards': 'AISHLYSBWVHWU', 'result': 'Pass'}, {'add': 'M', 'remove': 'U', 'cards': 'AISHLYSBWVHWM', 'result': 'Pass'}, {'add': 'V', 'remove': 'Y', 'cards': 'AISHLSBWVHWMV', 'result': 'Pass'}, {'add': 'C', 'remove': 'S', 'cards': 'AIHLSBWVHWMVC', 'result': 'Chi'}, {'add': 'K', 'remove': 'H', 'cards': 'AILSBWVHWMVCK', 'result': 'Chi'}, {'add': 'V', 'remove': 'M', 'cards': 'AILSBWVHWVCKV', 'result': 'Peng'}, {'add': 'W', 'remove': 'W', 'cards': 'AILSBVHWVCKVW', 'result': 'Peng'}, {'add': 'W', 'remove': 'A', 'cards': 'ILSBVHWVCKVWW', 'result': 'Peng'}], 'solution': 'Peng', 'difficulty': {'num_rounds': 10}}

````

### manipulate_matrix
Generates Manipulate Matrix exercises with configurable difficulty

Default configuration:
```python
min_rows = 2
min_cols = 2
max_rows = 10
max_cols = 10
min_transforms = 1
max_transforms = 10
w_rotate = 1
w_hmirror = 1
w_vmirror = 1
w_dmirror = 1
w_cmirror = 1
w_map = 1
w_crop = 1
w_remove_every_nth_row = 1
w_remove_every_nth_col = 1
w_zero_divisible = 1
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: For the following matrix:
4 3
3 2
1 8

Perform the following series of operations in order:
- Identity transformation, i.e. no change
- Map each occurrence of 0 to 9
- Rotate the matrix 180 degrees

Answer: 8 1
2 3
3 4
Metadata: {'matrix': [[4, 3], [3, 2], [1, 8]], 'solution': [[8, 1], [2, 3], [3, 4]], 'operations': [{'transform': 'map', 'from': 0, 'to': 9, 'instruction': '- Map each occurrence of 0 to 9'}, {'transform': 'rotate', 'degrees': '180', 'instruction': '- Rotate the matrix 180 degrees'}], 'difficulty': {'rows': 3, 'cols': 2, 'num_transforms': 2}}

Example 2:
Question: For the following matrix:
2 7 5 1 7 9
7 9 0 8 6 9

Perform the following series of operations in order:
- Identity transformation, i.e. no change
- Set all elements divisible by 9 to zero
- Remove every 2-nd row (1-indexed)
- Mirror the matrix along the diagonal
- Rotate the matrix 90 degrees
- Mirror the matrix along the diagonal
- Rotate the matrix 360 degrees

Answer: 0
7
1
5
7
2
Metadata: {'matrix': [[2, 7, 5, 1, 7, 9], [7, 9, 0, 8, 6, 9]], 'solution': [[0], [7], [1], [5], [7], [2]], 'operations': [{'transform': 'zero_divisible', 'k': 9, 'instruction': '- Set all elements divisible by 9 to zero'}, {'transform': 'remove_every_nth_row', 'n': 2, 'instruction': '- Remove every 2-nd row (1-indexed)'}, {'transform': 'dmirror', 'instruction': '- Mirror the matrix along the diagonal'}, {'transform': 'rotate', 'degrees': '90', 'instruction': '- Rotate the matrix 90 degrees'}, {'transform': 'dmirror', 'instruction': '- Mirror the matrix along the diagonal'}, {'transform': 'rotate', 'degrees': '360', 'instruction': '- Rotate the matrix 360 degrees'}], 'difficulty': {'rows': 2, 'cols': 6, 'num_transforms': 6}}

Example 3:
Question: For the following matrix:
8 1 2 6 3 4 0 3 1 9
0 1 2 8 4 6 9 6 5 5
1 5 4 9 2 1 8 1 9 1
4 5 1 4 0 5 6 1 7 7
3 3 2 4 3 0 0 6 0 5
5 7 7 9 8 2 3 7 7 5
9 0 4 2 0 3 0 9 9 8
8 4 5 9 3 6 1 5 5 1

Perform the following series of operations in order:
- Identity transformation, i.e. no change
- Set all elements divisible by 1 to zero
- Remove every 2-nd column (1-indexed)
- Set all elements divisible by 8 to zero
- Horizontally mirror the matrix

Answer: 0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
Metadata: {'matrix': [[8, 1, 2, 6, 3, 4, 0, 3, 1, 9], [0, 1, 2, 8, 4, 6, 9, 6, 5, 5], [1, 5, 4, 9, 2, 1, 8, 1, 9, 1], [4, 5, 1, 4, 0, 5, 6, 1, 7, 7], [3, 3, 2, 4, 3, 0, 0, 6, 0, 5], [5, 7, 7, 9, 8, 2, 3, 7, 7, 5], [9, 0, 4, 2, 0, 3, 0, 9, 9, 8], [8, 4, 5, 9, 3, 6, 1, 5, 5, 1]], 'solution': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'operations': [{'transform': 'zero_divisible', 'k': 1, 'instruction': '- Set all elements divisible by 1 to zero'}, {'transform': 'remove_every_nth_col', 'n': 2, 'instruction': '- Remove every 2-nd column (1-indexed)'}, {'transform': 'zero_divisible', 'k': 8, 'instruction': '- Set all elements divisible by 8 to zero'}, {'transform': 'hmirror', 'instruction': '- Horizontally mirror the matrix'}], 'difficulty': {'rows': 8, 'cols': 10, 'num_transforms': 4}}

````

### maze
Generates mazes with guaranteed shortest path distance from start to goal
    within [min_dist, max_dist].

Default configuration:
```python
min_dist = 5
max_dist = 10
min_grid_size = 5
max_grid_size = 10
seed = 42
size = 50
```

Example tasks:
````
Example 1:
Question: Navigate from '3' (start) to 'z' (goal):

```
>>>>>>>>>
>eeee>e>>
>ee>>>>>>
>eeeeee>>
>e>ee>>e>
>>ez>3e>>
>eee>e>e>
>eeeee>e>
>>>>>>>>>
```
Legend: '>' = Wall, 'e' = Passage

What is the minimum number of steps to reach the goal?
Give only the number of steps as your final answer, no other text or formatting.
Answer: 6
Metadata: {'grid_size': 9, 'grid': ['>>>>>>>>>', '>eeee>e>>', '>ee>>>>>>', '>eeeeee>>', '>e>ee>>e>', '>>ez>3e>>', '>eee>e>e>', '>eeeee>e>', '>>>>>>>>>'], 'shortest_path_length': 6, 'start': '3', 'goal': 'z', 'wall': '>', 'path': 'e', 'difficulty': {'dist': 6, 'grid_size': 9}}

Example 2:
Question: Navigate from '`' (start) to 'i' (goal):

```
4444444
4AAAAi4
4A4A4A4
4A4AA44
44AAAA4
44A`444
4444444
```
Legend: '4' = Wall, 'A' = Passage

What is the minimum number of steps to reach the goal?
Give only the number of steps as your final answer, no other text or formatting.
Answer: 6
Metadata: {'grid_size': 7, 'grid': ['4444444', '4AAAAi4', '4A4A4A4', '4A4AA44', '44AAAA4', '44A`444', '4444444'], 'shortest_path_length': 6, 'start': '`', 'goal': 'i', 'wall': '4', 'path': 'A', 'difficulty': {'dist': 6, 'grid_size': 7}}

Example 3:
Question: Navigate from '(' (start) to '`' (goal):

```
QQQQQQQ
QQ%%%%Q
QQ`%Q%Q
Q%%Q%%Q
Q%%%Q%Q
Q%QQ%(Q
QQQQQQQ
```
Legend: 'Q' = Wall, '%' = Passage

What is the minimum number of steps to reach the goal?
Give only the number of steps as your final answer, no other text or formatting.
Answer: 8
Metadata: {'grid_size': 7, 'grid': ['QQQQQQQ', 'QQ%%%%Q', 'QQ`%Q%Q', 'Q%%Q%%Q', 'Q%%%Q%Q', 'Q%QQ%(Q', 'QQQQQQQ'], 'shortest_path_length': 8, 'start': '(', 'goal': '`', 'wall': 'Q', 'path': '%', 'difficulty': {'dist': 8, 'grid_size': 7}}

````

### mini_sudoku
Generates 4x4 sudoku puzzles with configurable difficulty

Default configuration:
```python
min_empty = 8
max_empty = 12
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: In 4x4 Mini Sudoku:
- Each row must contain each number from 1-4 exactly once
- Each column must contain each number 1-4 exactly once
- Each 2x2 subgrid must contain each number 1-4 exactly once
Solve this 4x4 Mini Sudoku puzzle:
4 _ _ _
_ 3 _ _
_ 1 3 _
_ _ _ _
Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.

Answer: 4 2 1 3
1 3 4 2
2 1 3 4
3 4 2 1
Metadata: {'puzzle': [[4, 0, 0, 0], [0, 3, 0, 0], [0, 1, 3, 0], [0, 0, 0, 0]], 'solution': [[4, 2, 1, 3], [1, 3, 4, 2], [2, 1, 3, 4], [3, 4, 2, 1]], 'num_empty': 12, 'difficulty': {'empty': 12}}

Example 2:
Question: In 4x4 Mini Sudoku:
- Each row must contain each number from 1-4 exactly once
- Each column must contain each number 1-4 exactly once
- Each 2x2 subgrid must contain each number 1-4 exactly once
Solve this 4x4 Mini Sudoku puzzle:
3 _ _ _
_ _ 4 _
4 2 _ _
_ _ _ 4
Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.

Answer: 3 4 1 2
2 1 4 3
4 2 3 1
1 3 2 4
Metadata: {'puzzle': [[3, 0, 0, 0], [0, 0, 4, 0], [4, 2, 0, 0], [0, 0, 0, 4]], 'solution': [[3, 4, 1, 2], [2, 1, 4, 3], [4, 2, 3, 1], [1, 3, 2, 4]], 'num_empty': 11, 'difficulty': {'empty': 11}}

Example 3:
Question: In 4x4 Mini Sudoku:
- Each row must contain each number from 1-4 exactly once
- Each column must contain each number 1-4 exactly once
- Each 2x2 subgrid must contain each number 1-4 exactly once
Solve this 4x4 Mini Sudoku puzzle:
_ _ _ _
1 3 4 _
3 _ 2 4
4 _ _ 1
Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.

Answer: 2 4 1 3
1 3 4 2
3 1 2 4
4 2 3 1
Metadata: {'puzzle': [[0, 0, 0, 0], [1, 3, 4, 0], [3, 0, 2, 4], [4, 0, 0, 1]], 'solution': [[2, 4, 1, 3], [1, 3, 4, 2], [3, 1, 2, 4], [4, 2, 3, 1]], 'num_empty': 8, 'difficulty': {'empty': 8}}

````

### modulo_grid
Generates ModuloGrid tasks

    This is an ARC-ish task for mathematical explanatory reasoning. It generates a binary grid based on a hidden
    mathematical function based around modulo division of a function based on the coordinates, then asks to fill
    in any gaps in the grid.

    The function used to determine the pattern can be based on sums, multiples, powers, and differences, then a
    constructed modulo matching a target function. Some patterns are obvious without knowing the underlying rule,
    some are very difficult. Pretty much all the parameters are configurable, so we are able to generate a
    good curriculum.

Default configuration:
```python
size_x = 20
size_y = 20
max_divisor = 20
max_target = 20
max_holes = 1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Identify the mathematical pattern which defines this grid, then use that pattern to fill in the question marks. Return the entire completed grid as your answer.

❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌❔❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
Answer: ❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
Metadata: {'divisor': 4, 'target': 1, 'operation': 'prod', 'difficulty': {'holes': 1, 'size_x': 20, 'size_y': 20}}

Example 2:
Question: Identify the mathematical pattern which defines this grid, then use that pattern to fill in the question marks. Return the entire completed grid as your answer.

❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅
❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌
❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌
❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌
✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❔
❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌
Answer: ❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅
❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌
❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌
❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌
✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌
Metadata: {'divisor': 15, 'target': 12, 'operation': 'sum', 'difficulty': {'holes': 1, 'size_x': 20, 'size_y': 20}}

Example 3:
Question: Identify the mathematical pattern which defines this grid, then use that pattern to fill in the question marks. Return the entire completed grid as your answer.

❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌
❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌
❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❔❌
❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌
❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌
❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌
❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌
❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅
❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌
✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌
❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌
❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌
❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌
❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌
❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌
❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌
❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌
Answer: ❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌
❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌
❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌
❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌❌
❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌❌
❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌❌
❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅❌
❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌✅
❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌❌
✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌❌
❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌❌
❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌❌
❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌❌
❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌❌
❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌❌
❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅❌
❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌✅
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌✅❌
Metadata: {'divisor': 10, 'target': 1, 'operation': 'diff', 'difficulty': {'holes': 1, 'size_x': 20, 'size_y': 20}}

````

### n_queens
Generates N Queens puzzles with configurable difficulty

Default configuration:
```python
n = 8
min_remove = 1
max_remove = 7
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

You can place a queen by replacing an underscore (_) with a Q.

Your output should be also a board in the same format as the input, with queens placed on the board by replacing underscores with the letter Q.

Given the below board of size 8 x 8 your job is to place 1 queen(s) on the board such that no two queens attack each other.
_ _ _ _ _ _ Q _
_ Q _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ Q
_ _ _ _ Q _ _ _
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _

Answer: _ _ _ _ _ _ Q _
_ Q _ _ _ _ _ _
_ _ _ Q _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ _ _ _ _ Q
_ _ _ _ Q _ _ _
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _
Metadata: {'puzzle': [['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], 'solutions': [[['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']]], 'num_removed': 1, 'valid_answers': ['_ _ _ _ _ _ Q _\n_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ _ Q\n_ _ _ _ Q _ _ _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _'], 'difficulty': {'n': 8, 'num_removed': 1}}

Example 2:
Question: Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

You can place a queen by replacing an underscore (_) with a Q.

Your output should be also a board in the same format as the input, with queens placed on the board by replacing underscores with the letter Q.

Given the below board of size 8 x 8 your job is to place 3 queen(s) on the board such that no two queens attack each other.
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ Q _ _
_ _ _ _ _ _ _ Q
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ Q _
_ _ _ _ Q _ _ _

Answer: _ Q _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ Q _ _
_ _ _ _ _ _ _ Q
_ _ Q _ _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ _ _ _ Q _
_ _ _ _ Q _ _ _
Metadata: {'puzzle': [['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', '_', '_', 'Q', '_', '_', '_']], 'solutions': [[['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', '_', '_', 'Q', '_', '_', '_']]], 'num_removed': 3, 'valid_answers': ['_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\n_ _ _ _ _ Q _ _\n_ _ _ _ _ _ _ Q\n_ _ Q _ _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ Q _\n_ _ _ _ Q _ _ _'], 'difficulty': {'n': 8, 'num_removed': 3}}

Example 3:
Question: Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

You can place a queen by replacing an underscore (_) with a Q.

Your output should be also a board in the same format as the input, with queens placed on the board by replacing underscores with the letter Q.

Given the below board of size 8 x 8 your job is to place 5 queen(s) on the board such that no two queens attack each other.
_ _ _ _ _ _ _ _
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ Q _ _

Answer: _ _ _ _ Q _ _ _
_ Q _ _ _ _ _ _
_ _ _ _ _ _ _ Q
Q _ _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ _ Q _
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _
Metadata: {'puzzle': [['_', '_', '_', '_', '_', '_', '_', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], 'solutions': [[['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], [['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']], [['_', '_', '_', '_', '_', '_', '_', 'Q'], ['_', 'Q', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'Q', '_', '_', '_', '_'], ['Q', '_', '_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_', 'Q', '_'], ['_', '_', '_', '_', 'Q', '_', '_', '_'], ['_', '_', 'Q', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', 'Q', '_', '_']]], 'num_removed': 5, 'valid_answers': ['_ _ _ _ Q _ _ _\n_ Q _ _ _ _ _ _\n_ _ _ _ _ _ _ Q\nQ _ _ _ _ _ _ _\n_ _ _ Q _ _ _ _\n_ _ _ _ _ _ Q _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _', '_ _ _ _ _ _ Q _\n_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ _ Q\n_ _ _ _ Q _ _ _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _', '_ _ _ _ _ _ _ Q\n_ Q _ _ _ _ _ _\n_ _ _ Q _ _ _ _\nQ _ _ _ _ _ _ _\n_ _ _ _ _ _ Q _\n_ _ _ _ Q _ _ _\n_ _ Q _ _ _ _ _\n_ _ _ _ _ Q _ _'], 'difficulty': {'n': 8, 'num_removed': 5}}

````

### needle_haystack
Generates "Needle in a Haystack tasks

Default configuration:
```python
min_num_statements = 10
max_num_statements = 100
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Caolain is neutral toward music. Alexx desires writing novels. Jake bears boxing. Harold gripes about dusting the furniture. Frederick disdains ironing the curtains. Cooper enjoys astronomy hobby. Caiden-Paul applauds all-terrain vehicles. Shayne delights in politics. Bradyn accepts artificial intelligence. Tyrnan supports climbing. Michal yearns for acting. Alvin deifies penguins. Allen relishes sailing. Brooke overlooks archery. Flynn prizes cleaning the patio. Grady can’t bear brewing beer. Rio ridicules acting. Wen is committed to emptying the dishwasher. Alfy execrates weeding the garden. Sweyn deifies bats. Emlyn laments bats. Shayan is passionate about snowboarding. Mehmet idolizes bird photography. Francis pines octopuses. Nikash worships ice skating. Tymom fancies motorcycles. Jaosha rejects balance. Abdur celebrates anime. Darryn bemoans logic. Michee revels in cleaning the ceiling fan. Khaleel worships trains. Jamie rails against the color amber. Daragh exults in astronomy. Finlay scoffs at minibikes. Kenyon desires collecting postcards. Caiden worships cocktails. Brodie reviles writing novels. Linton extols virtual reality. Bryson covets playing volleyball. Kyan begrudges listening to jazz. Kieran-Scott disapproves of collecting postcards. Willum esteems indie films. Isaa is addicted to ballet dancing. Arafat finds pleasure in triathlon. Oluwafemi disapproves of astronomy hobby. Seamas is keen on diving. Cian blasts playing the banjo. Liam-Stephen loathes the color sapphire. Bilal shrugs off playing the accordion. Sol is crazy about hip-hop dancing. Joaquin finds joy in dancing tango. Zakir idolizes rowing. Kym accepts raking the leaves. Timothy fancies beatboxing. Efe extols bats. Bailley pines for zebras. Areeb adores popcorn. Geordie deifies street art. Jerome idolizes geology. Blaike disdains landscape photography. Graeme dotes bulldozers. Caelen damns fruit salad. Lionel yearns ice cream. Jan disapproves of classic literature. Oakley appreciates reading science fiction. Laird delights in pie. Dawid ignores climbing. Leigham derides cleaning the garage. Kriss is devoted to playing violin. Farhaan tolerates skiing. Issiaka extols kayaks. Baron finds pleasure in machine learning. Yaseen glorifies tea. Derren brushes off recycling. Saunders delights in lobsters. Harold is crazy about baking cakes. Daymian abhors the color chartreuse. Orrick finds fulfillment in the color peach. Lincon rejects listening to folk music. Jacky disdains monkeys. Stewarty dismisses cows. Forgan is obsessed with manga. Kayden thrives on cryptocurrency. Ashton craves imagination. Ghyll disdains fencing. Amaan reveres electric cars. Kajetan laments playing percussion. Malakai loves burger. Izaak exults in playing bowling. Azedine regrets tacos. Fawaz regrets performing stand-up comedy. 
Who glorifies tea? Reply only with a name.
Answer: Yaseen
Metadata: {'question': 'Who glorifies tea? Reply only with a name.', 'difficulty': {'num_statements': 91}}

Example 2:
Question: Jazz adores trail running. Craig eschews ballet dancing. Orrin resents wolves. Leigh adores playing ping pong. Bryn spurns washing the dishes. Nyah dotes foxes. Vuyolwethu finds fulfillment in DJing. Rhoridh rails against baking cakes. Yaseen idolizes goats. Ajayraj lusts after visiting theme parks. Rooke damns building model airplanes. Morton approves of bird photography. Tiarnan curses trucks. Lennon endorses deer. 
Who lusts after visiting theme parks? Reply only with a name.
Answer: Ajayraj
Metadata: {'question': 'Who lusts after visiting theme parks? Reply only with a name.', 'difficulty': {'num_statements': 14}}

Example 3:
Question: Rufus mocks geocaching. Sharland yearns for the color yellow. Cejay yearns exploring caves. Diarmuid reveres limousines. Lincon exults resilience. Gareth ridicules playing board games. Jerome gripes about off-road vehicles. Aliyaan loves courage. Gabriel worships trucks. Cejay craves triathlon. Taylor-Jay detests off-road vehicles. Abu adores determination. Caedyn spurns pie. Darien is indifferent to resilience. Ronnie scorns all-terrain vehicles. Josan tolerates playing saxophone. Liam scorns playing cricket. Tyson longs for scorpions. Marc-Anthony ignores making coffee. Kayne bears trail running. Kurtis blasts creativity. Beau appreciates racing cars. Kerr laments the color khaki. Jayden-Paul relishes mopping the floor. Zak appreciates metaphysics. Darroch detests beauty. Carlo regrets building model cars. Rogan stomachs listening to folk music. Baley execrates omelettes. Tyler-Jay despises washing the dishes. Bruno fancies popcorn. Jacky puts up with zoology. Kajetan mocks cleaning the oven. Calley desires the color fuchsia. Zishan supports optimism. Jeronimo can’t bear vacuuming the floor. Amolpreet mocks roller skating. Kierin regrets metaphysics. Loudon approves of ducks. Brydon despises camels. Prinay eschews roller skating. Precious reveres coffee. Edison damns playing cricket. Eason yearns ants. Codey lusts after the color ruby. Ian revels in virtual reality. Hashim respects the color blue. Armaan derides performing magic. Arafat revels in canoeing. Murdo glories in pizza. Anesu exalts tea. Kedrick can’t stand whales. Layton rails against cleaning the carpets. Peirce approves of stir-fry. Oban loves filmmaking. Tyra idolizes balloons. Shadow extols playing violin. Damon shuns salad. Grahame esteems ironing clothes. Ralph desires ferries. Ohran abides vans. Kimi reviles cars. 
Who despises camels? Reply only with a name.
Answer: Brydon
Metadata: {'question': 'Who despises camels? Reply only with a name.', 'difficulty': {'num_statements': 62}}

````

### number_filtering
Generates number filtering tasks

Default configuration:
```python
min_numbers = 3
max_numbers = 10
min_decimals = 0
max_decimals = 4
min_value = -100.0
max_value = 100.0
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Keep all numbers larger than -90 in this list: ['-95.00', '-51.0', '47.2942', '-82.612']
Return the new list in the same format.
Answer: ['-51.0', '47.2942', '-82.612']
Metadata: {'original_numbers': ['-95.00', '-51.0', '47.2942', '-82.612'], 'filter_value': '-90', 'operation': 'keep_larger', 'result': ['-51.0', '47.2942', '-82.612'], 'difficulty': {'numbers': 4, 'decimals': (0, 4), 'value': (-100.0, 100.0)}}

Example 2:
Question: Remove all numbers larger than 18.236 in this list: ['-42.8', '91.88', '34']
Return the new list in the same format.
Answer: ['-42.8']
Metadata: {'original_numbers': ['-42.8', '91.88', '34'], 'filter_value': '18.236', 'operation': 'remove_larger', 'result': ['-42.8'], 'difficulty': {'numbers': 3, 'decimals': (0, 4), 'value': (-100.0, 100.0)}}

Example 3:
Question: Keep all numbers larger than 19.8962 in this list: ['4', '-64.7', '-42.1', '-77', '-79.9640', '37.76', '38.702', '18.20', '-28.34']
Return the new list in the same format.
Answer: ['37.76', '38.702']
Metadata: {'original_numbers': ['4', '-64.7', '-42.1', '-77', '-79.9640', '37.76', '38.702', '18.20', '-28.34'], 'filter_value': '19.8962', 'operation': 'keep_larger', 'result': ['37.76', '38.702'], 'difficulty': {'numbers': 9, 'decimals': (0, 4), 'value': (-100.0, 100.0)}}

````

### number_format
Generates Count Bits exercises with configurable difficulty

Default configuration:
```python
min_num_candidates = 2
max_num_candidates = 5
min_n = 1000
max_n = 1000000000
max_delta = 10.0
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your task is to pick the largest/smallest number out of several options.

Your output should be only the number of interest.

Now, pick the largest number of the following candidates: 25011730.212000 25011725.713000

Answer: 25011730.212
Metadata: {'candidates': [25011730.212, 25011725.713], 'solution': 25011730.212, 'formatted_candidates': ['25011730.212000', '25011725.713000'], 'size': 'largest', 'difficulty': {'num_candidates': 2, 'n': (1000, 1000000000), 'min_delta': 10.0}}

Example 2:
Question: Your task is to pick the largest/smallest number out of several options.

Your output should be only the number of interest.

Now, pick the largest number of the following candidates: 286,084,894.213 286,084,899.467

Answer: 286084899.467
Metadata: {'candidates': [286084894.213, 286084899.467], 'solution': 286084899.467, 'formatted_candidates': ['286,084,894.213', '286,084,899.467'], 'size': 'largest', 'difficulty': {'num_candidates': 2, 'n': (1000, 1000000000), 'min_delta': 10.0}}

Example 3:
Question: Your task is to pick the largest/smallest number out of several options.

Your output should be only the number of interest.

Now, pick the largest number of the following candidates: 520020968.942000 520020972.974000 5.200209612750000e+08 520020966.533000 520020964.733000

Answer: 520020972.974
Metadata: {'candidates': [520020968.942, 520020972.974, 520020961.275, 520020966.533, 520020964.733], 'solution': 520020972.974, 'formatted_candidates': ['520020968.942000', '520020972.974000', '5.200209612750000e+08', '520020966.533000', '520020964.733000'], 'size': 'largest', 'difficulty': {'num_candidates': 5, 'n': (1000, 1000000000), 'min_delta': 10.0}}

````

### number_sequence
Generates number sequence completion tasks with dynamic pattern generation

Default configuration:
```python
min_terms = 4
max_terms = 8
min_value = -100
max_value = 100
max_complexity = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: 3, 6, 12, 24, 48, 96, 192, 384, ?
Answer: 768
Metadata: {'rule': 'double', 'complexity': 3, 'sequence': [3, 6, 12, 24, 48, 96, 192, 384, 768]}

Example 2:
Question: 8, 14, 20, 26, 32, 38, 44, ?
Answer: 50
Metadata: {'rule': 'add 6', 'complexity': 1, 'sequence': [8, 14, 20, 26, 32, 38, 44, 50]}

Example 3:
Question: 8, 4, 2, 1, 0, 0, 0, ?
Answer: 0
Metadata: {'rule': 'halve', 'complexity': 2, 'sequence': [8, 4, 2, 1, 0, 0, 0, 0]}

````

### number_sorting
Generates number sorting tasks

Default configuration:
```python
min_numbers = 3
max_numbers = 10
min_decimals = 0
max_decimals = 2
min_value = -100.0
max_value = 100.0
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Sort these numbers in ascending order: -95.0, -51, 47.29, 48.13
Please follow the instruction below:
## 1. Let all your answers be a list of numbers. Instead of reporting your answer as -69, -13, 1, 7, 11, 43, 59, 61, use ['-69', '-13', '1', '7', '11', '43', '59', '61'] instead
## 2. Convert all numbers in the square brackets as strings. For example, ['-69', '-13', '1', '7', '11', '43', '59', '61']

Answer: ['-95.0', '-51.0', '47.3', '48.1']
Metadata: {'original_numbers': ['-95.0', '-51', '47.29', '48.13'], 'direction': 'ascending', 'sorted_numbers': ['-95.0', '-51.0', '47.3', '48.1'], 'difficulty': {'numbers': 4, 'decimals': (0, 2), 'value': (-100.0, 100.0)}}

Example 2:
Question: Sort these numbers in descending order: -43, 91.9, 34
Please follow the instruction below:
## 1. Let all your answers be a list of numbers. Instead of reporting your answer as -69, -13, 1, 7, 11, 43, 59, 61, use ['-69', '-13', '1', '7', '11', '43', '59', '61'] instead
## 2. Convert all numbers in the square brackets as strings. For example, ['-69', '-13', '1', '7', '11', '43', '59', '61']

Answer: ['92', '34', '-43']
Metadata: {'original_numbers': ['-43', '91.9', '34'], 'direction': 'descending', 'sorted_numbers': ['92', '34', '-43'], 'difficulty': {'numbers': 3, 'decimals': (0, 2), 'value': (-100.0, 100.0)}}

Example 3:
Question: Sort these numbers in descending order: 4.00, 72, -24.1, -94, 14, -68.66, 37.8, 38.7, 18.20
Please follow the instruction below:
## 1. Let all your answers be a list of numbers. Instead of reporting your answer as -69, -13, 1, 7, 11, 43, 59, 61, use ['-69', '-13', '1', '7', '11', '43', '59', '61'] instead
## 2. Convert all numbers in the square brackets as strings. For example, ['-69', '-13', '1', '7', '11', '43', '59', '61']

Answer: ['72.00', '38.70', '37.80', '18.20', '14.00', '4.00', '-24.10', '-68.66', '-94.00']
Metadata: {'original_numbers': ['4.00', '72', '-24.1', '-94', '14', '-68.66', '37.8', '38.7', '18.20'], 'direction': 'descending', 'sorted_numbers': ['72.00', '38.70', '37.80', '18.20', '14.00', '4.00', '-24.10', '-68.66', '-94.00'], 'difficulty': {'numbers': 9, 'decimals': (0, 2), 'value': (-100.0, 100.0)}}

````

### palindrome_generation
Generates a set of letters that can be assembled into a palindrome.

Default configuration:
```python
min_length = 3
max_length = 10
seed = 42
size = 50
```

Example tasks:
````
Example 1:
Question: Your task is, given a list of letters, to form a valid palindrome.

A palindrome is a phrase that reads the same forwards and backwards.

If there are multiple possible answers, only respond with one of them. You must use all the letters provided.

Your output should be a single string, with no spaces or punctuation.

Now, form a valid palindrome using the following letters: h, a, h, a

Answer: ahha
Metadata: {'letters': ['h', 'a', 'h', 'a'], 'generated_palindrome': 'ahha', 'difficulty': {'length': 4}}

Example 2:
Question: Your task is, given a list of letters, to form a valid palindrome.

A palindrome is a phrase that reads the same forwards and backwards.

If there are multiple possible answers, only respond with one of them. You must use all the letters provided.

Your output should be a single string, with no spaces or punctuation.

Now, form a valid palindrome using the following letters: h, y, h

Answer: hyh
Metadata: {'letters': ['h', 'y', 'h'], 'generated_palindrome': 'hyh', 'difficulty': {'length': 3}}

Example 3:
Question: Your task is, given a list of letters, to form a valid palindrome.

A palindrome is a phrase that reads the same forwards and backwards.

If there are multiple possible answers, only respond with one of them. You must use all the letters provided.

Your output should be a single string, with no spaces or punctuation.

Now, form a valid palindrome using the following letters: n, j, n, j, d, j, s, s, d

Answer: nsdjjjdsn
Metadata: {'letters': ['n', 'j', 'n', 'j', 'd', 'j', 's', 's', 'd'], 'generated_palindrome': 'nsdjjjdsn', 'difficulty': {'length': 9}}

````

### palindrome_partitioning
Generates Palindrome Partitioning exercises with configurable difficulty

Default configuration:
```python
min_string_len = 5
max_string_len = 15
min_substring_palindrome_len = 1
max_substring_palindrome_len = 5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Your output should be a list of lists, where each list represents a palindrome partition, e.g. [["a","a","b"],["aa","b"]].

Partition the following string into palindromes: agegvckakcgnnrw

Answer: [["a", "g", "e", "g", "v", "c", "k", "a", "k", "c", "g", "n", "n", "r", "w"], ["a", "g", "e", "g", "v", "c", "k", "a", "k", "c", "g", "nn", "r", "w"], ["a", "g", "e", "g", "v", "c", "kak", "c", "g", "n", "n", "r", "w"], ["a", "g", "e", "g", "v", "c", "kak", "c", "g", "nn", "r", "w"], ["a", "g", "e", "g", "v", "ckakc", "g", "n", "n", "r", "w"], ["a", "g", "e", "g", "v", "ckakc", "g", "nn", "r", "w"], ["a", "geg", "v", "c", "k", "a", "k", "c", "g", "n", "n", "r", "w"], ["a", "geg", "v", "c", "k", "a", "k", "c", "g", "nn", "r", "w"], ["a", "geg", "v", "c", "kak", "c", "g", "n", "n", "r", "w"], ["a", "geg", "v", "c", "kak", "c", "g", "nn", "r", "w"], ["a", "geg", "v", "ckakc", "g", "n", "n", "r", "w"], ["a", "geg", "v", "ckakc", "g", "nn", "r", "w"]]
Metadata: {'string': 'agegvckakcgnnrw', 'solution': [['a', 'g', 'e', 'g', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'nn', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'c', 'kak', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'c', 'kak', 'c', 'g', 'nn', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'ckakc', 'g', 'n', 'n', 'r', 'w'], ['a', 'g', 'e', 'g', 'v', 'ckakc', 'g', 'nn', 'r', 'w'], ['a', 'geg', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'geg', 'v', 'c', 'k', 'a', 'k', 'c', 'g', 'nn', 'r', 'w'], ['a', 'geg', 'v', 'c', 'kak', 'c', 'g', 'n', 'n', 'r', 'w'], ['a', 'geg', 'v', 'c', 'kak', 'c', 'g', 'nn', 'r', 'w'], ['a', 'geg', 'v', 'ckakc', 'g', 'n', 'n', 'r', 'w'], ['a', 'geg', 'v', 'ckakc', 'g', 'nn', 'r', 'w']], 'difficulty': {'string_len': 15}}

Example 2:
Question: Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Your output should be a list of lists, where each list represents a palindrome partition, e.g. [["a","a","b"],["aa","b"]].

Partition the following string into palindromes: sesjj

Answer: [["s", "e", "s", "j", "j"], ["s", "e", "s", "jj"], ["ses", "j", "j"], ["ses", "jj"]]
Metadata: {'string': 'sesjj', 'solution': [['s', 'e', 's', 'j', 'j'], ['s', 'e', 's', 'jj'], ['ses', 'j', 'j'], ['ses', 'jj']], 'difficulty': {'string_len': 5}}

Example 3:
Question: Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Your output should be a list of lists, where each list represents a palindrome partition, e.g. [["a","a","b"],["aa","b"]].

Partition the following string into palindromes: owfwofaafsd

Answer: [["o", "w", "f", "w", "o", "f", "a", "a", "f", "s", "d"], ["o", "w", "f", "w", "o", "f", "aa", "f", "s", "d"], ["o", "w", "f", "w", "o", "faaf", "s", "d"], ["o", "wfw", "o", "f", "a", "a", "f", "s", "d"], ["o", "wfw", "o", "f", "aa", "f", "s", "d"], ["o", "wfw", "o", "faaf", "s", "d"], ["owfwo", "f", "a", "a", "f", "s", "d"], ["owfwo", "f", "aa", "f", "s", "d"], ["owfwo", "faaf", "s", "d"]]
Metadata: {'string': 'owfwofaafsd', 'solution': [['o', 'w', 'f', 'w', 'o', 'f', 'a', 'a', 'f', 's', 'd'], ['o', 'w', 'f', 'w', 'o', 'f', 'aa', 'f', 's', 'd'], ['o', 'w', 'f', 'w', 'o', 'faaf', 's', 'd'], ['o', 'wfw', 'o', 'f', 'a', 'a', 'f', 's', 'd'], ['o', 'wfw', 'o', 'f', 'aa', 'f', 's', 'd'], ['o', 'wfw', 'o', 'faaf', 's', 'd'], ['owfwo', 'f', 'a', 'a', 'f', 's', 'd'], ['owfwo', 'f', 'aa', 'f', 's', 'd'], ['owfwo', 'faaf', 's', 'd']], 'difficulty': {'string_len': 11}}

````

### polynomial_equations
Generates random polynomial equations of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we solve "polynomial_expr = 0" using Sympy.
    - The solution may be real or complex; we filter real solutions by default for simplicity.

Default configuration:
```python
min_terms = 2
max_terms = 4
min_value = 1
max_value = 100
min_degree = 1
max_degree = 3
operators = ('+', '-')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the real value(s) of w in the equation: -127*w = 0
In solving equations, please follow these instructions:
1. Provide all answers as comma-separated decimal values. For example: "-0.3773, 0.4005"
2. For solutions that can be expressed in exact form (like "u = 2 + sqrt(4560)/172" and "u = 2 - sqrt(4560)/172"), convert them to decimal form in your final answer.
3. If there are no real values that satisfy the equation, report your answer as an empty string: ""
4. Format your answer based on the number of solutions:
   - For 1 solution: a single decimal number
   - For 2 solutions: two comma-separated decimal numbers
   - For 3 or more solutions: all values as comma-separated decimal numbers
5. Round all decimal values to 4 decimal places (rounding down when the 5th decimal place is 5 or greater).

Answer: 0.0
Metadata: {'polynomial_expr': '-127*w', 'variable': 'w', 'degree': 1, 'real_solutions': [0.0], 'difficulty': {'terms': 2, 'degree': 1}}

Example 2:
Question: Determine the real value(s) of b that satisfies: 86*b**2 - 2*b - 13 = 0
In solving equations, please follow these instructions:
1. Provide all answers as comma-separated decimal values. For example: "-0.3773, 0.4005"
2. For solutions that can be expressed in exact form (like "u = 2 + sqrt(4560)/172" and "u = 2 - sqrt(4560)/172"), convert them to decimal form in your final answer.
3. If there are no real values that satisfy the equation, report your answer as an empty string: ""
4. Format your answer based on the number of solutions:
   - For 1 solution: a single decimal number
   - For 2 solutions: two comma-separated decimal numbers
   - For 3 or more solutions: all values as comma-separated decimal numbers
5. Round all decimal values to 4 decimal places (rounding down when the 5th decimal place is 5 or greater).

Answer: -0.3773, 0.4006
Metadata: {'polynomial_expr': '86*b**2 - 2*b - 13', 'variable': 'b', 'degree': 2, 'real_solutions': [-0.3773, 0.4006], 'difficulty': {'terms': 4, 'degree': 2}}

Example 3:
Question: Determine the real value(s) of p that satisfies: 71*p**3 - 2*p - 29 = 0
In solving equations, please follow these instructions:
1. Provide all answers as comma-separated decimal values. For example: "-0.3773, 0.4005"
2. For solutions that can be expressed in exact form (like "u = 2 + sqrt(4560)/172" and "u = 2 - sqrt(4560)/172"), convert them to decimal form in your final answer.
3. If there are no real values that satisfy the equation, report your answer as an empty string: ""
4. Format your answer based on the number of solutions:
   - For 1 solution: a single decimal number
   - For 2 solutions: two comma-separated decimal numbers
   - For 3 or more solutions: all values as comma-separated decimal numbers
5. Round all decimal values to 4 decimal places (rounding down when the 5th decimal place is 5 or greater).

Answer: 0.7546
Metadata: {'polynomial_expr': '71*p**3 - 2*p - 29', 'variable': 'p', 'degree': 3, 'real_solutions': [0.7546], 'difficulty': {'terms': 4, 'degree': 3}}

````

### polynomial_multiplication
Generates [min_polynomials, max_polynomials] random polynomials of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we find "F = P_0 * ... * P_1" using Sympy.

Default configuration:
```python
min_terms = 2
max_terms = 4
min_value = 1
max_value = 100
min_degree = 0
max_degree = 3
min_polynomials = 2
max_polynomials = 3
variables = ('x', 'y', 'z')
allow_cross_variable_product = False
allow_multivariate_polynomials = False
operators = ('+', '-')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Calculate the following: (18*z**3 - 95*z)*(78*z**3 - 104*z**2 - 12)
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.

Answer: 1404*z**6 - 1872*z**5 - 7410*z**4 + 9664*z**3 + 1140*z
Metadata: {'polynomial_expr': '(18*z**3 - 95*z)*(78*z**3 - 104*z**2 - 12)', 'variables': [z], 'difficulty': {'min_terms': 2, 'max_terms': 4, 'min_value': 1, 'max_value': 100, 'min_degree': 0, 'max_degree': 3, 'min_polynomials': 2, 'max_polynomials': 3}}

Example 2:
Question: Simplify this expression: (-49*x**3 + 8*x**2 - 163)*(8*x**3 - 49*x**2 + 77*x)*(98*x**3 + 16*x**2 + 74*x)
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.

Answer: -38416*x**9 + 235298*x**8 - 397738*x**7 + 48346*x**6 + 463508*x**5 - 1153118*x**4 + 390222*x**3 - 928774*x**2
Metadata: {'polynomial_expr': '(-49*x**3 + 8*x**2 - 163)*(8*x**3 - 49*x**2 + 77*x)*(98*x**3 + 16*x**2 + 74*x)', 'variables': [x], 'difficulty': {'min_terms': 2, 'max_terms': 4, 'min_value': 1, 'max_value': 100, 'min_degree': 0, 'max_degree': 3, 'min_polynomials': 2, 'max_polynomials': 3}}

Example 3:
Question: Calculate the following: (49*y**2 + 21)*(29*y**3 - 49*y)
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.

Answer: 1421*y**5 - 1792*y**3 - 1029*y
Metadata: {'polynomial_expr': '(49*y**2 + 21)*(29*y**3 - 49*y)', 'variables': [y], 'difficulty': {'min_terms': 2, 'max_terms': 4, 'min_value': 1, 'max_value': 100, 'min_degree': 0, 'max_degree': 3, 'min_polynomials': 2, 'max_polynomials': 3}}

````

### pool_matrix
Generates Pool Matrix exercises with configurable difficulty

Default configuration:
```python
min_rows = 2
max_rows = 10
min_cols = 2
max_cols = 10
min_pool_size = 1
max_pool_size = 3
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your job is to perform max/average pooling on the given matrix.
The stride is equal to the kernel size, meaning there is no overlap between the pooling regions.

Your output should be a matrix in the same format as the input matrix.
The output matrix is smaller than the input matrix when the kernel size is greater than 1, and its elements may be floating-point numbers.
Give elements in the output matrix correct to 2 decimal places.

Perform max pooling on the following matrix with a kernel size of 3:
6 3
7 4
6 9

Answer: 9
Metadata: {'matrix': [[6, 3], [7, 4], [6, 9]], 'pool_type': 'max', 'pool_size': 3, 'solution': [[9]], 'difficulty': {'rows': 3, 'cols': 2, 'pool_size': 3}}

Example 2:
Question: Your job is to perform max/average pooling on the given matrix.
The stride is equal to the kernel size, meaning there is no overlap between the pooling regions.

Your output should be a matrix in the same format as the input matrix.
The output matrix is smaller than the input matrix when the kernel size is greater than 1, and its elements may be floating-point numbers.
Give elements in the output matrix correct to 2 decimal places.

Perform average pooling on the following matrix with a kernel size of 3:
4 0 1 5 0 3
1 2 7 0 3 2

Answer: 2.5 2.17
Metadata: {'matrix': [[4, 0, 1, 5, 0, 3], [1, 2, 7, 0, 3, 2]], 'pool_type': 'average', 'pool_size': 3, 'solution': [[2.5, 2.1666666666666665]], 'difficulty': {'rows': 2, 'cols': 6, 'pool_size': 3}}

Example 3:
Question: Your job is to perform max/average pooling on the given matrix.
The stride is equal to the kernel size, meaning there is no overlap between the pooling regions.

Your output should be a matrix in the same format as the input matrix.
The output matrix is smaller than the input matrix when the kernel size is greater than 1, and its elements may be floating-point numbers.
Give elements in the output matrix correct to 2 decimal places.

Perform average pooling on the following matrix with a kernel size of 3:
4 3 1 3 0 4 3 8 7 7
6 9 3 7 3 3 6 5 4 5
9 1 8 7 4 5 3 0 4 9
2 8 8 6 2 0 3 4 8 3
2 2 1 2 2 9 8 1 8 9
4 2 4 6 7 5 5 6 2 5
1 8 9 1 8 0 9 3 5 9
5 0 8 0 4 2 9 7 6 6

Answer: 4.89 4.0 4.44 7.0
3.67 4.33 5.0 5.67
5.17 2.5 6.5 7.5
Metadata: {'matrix': [[4, 3, 1, 3, 0, 4, 3, 8, 7, 7], [6, 9, 3, 7, 3, 3, 6, 5, 4, 5], [9, 1, 8, 7, 4, 5, 3, 0, 4, 9], [2, 8, 8, 6, 2, 0, 3, 4, 8, 3], [2, 2, 1, 2, 2, 9, 8, 1, 8, 9], [4, 2, 4, 6, 7, 5, 5, 6, 2, 5], [1, 8, 9, 1, 8, 0, 9, 3, 5, 9], [5, 0, 8, 0, 4, 2, 9, 7, 6, 6]], 'pool_type': 'average', 'pool_size': 3, 'solution': [[4.888888888888889, 4.0, 4.444444444444445, 7.0], [3.6666666666666665, 4.333333333333333, 5.0, 5.666666666666667], [5.166666666666667, 2.5, 6.5, 7.5]], 'difficulty': {'rows': 8, 'cols': 10, 'pool_size': 3}}

````

### power_function
Generates Power Function exercises with configurable difficulty

Default configuration:
```python
min_base = -1000.0
max_base = 1000.0
min_exponent = 0
max_exponent = 8
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your task is to compute an exponentiation of a number.

Compute 278.8536^0. Return your final answer correct to 3 significant figures.
Provide your answer in scientific notation using 'e' notation (e.g., 1.23e+4).

Answer: 1.0
Metadata: {'base': 278.8536, 'exponent': 0, 'solution': 1.0, 'difficulty': {'exponent': 0}}

Example 2:
Question: Your task is to compute an exponentiation of a number.

Compute -922.8963^2. Return your final answer correct to 3 significant figures.
Provide your answer in scientific notation using 'e' notation (e.g., 1.23e+4).

Answer: 851737.58055369
Metadata: {'base': -922.8963, 'exponent': 2, 'solution': 851737.58055369, 'difficulty': {'exponent': 2}}

Example 3:
Question: Your task is to compute an exponentiation of a number.

Compute -182.9282^8. Return your final answer correct to 3 significant figures.
Provide your answer in scientific notation using 'e' notation (e.g., 1.23e+4).

Answer: 1.2538491439703905e+18
Metadata: {'base': -182.9282, 'exponent': 8, 'solution': 1.2538491439703905e+18, 'difficulty': {'exponent': 8}}

````

### prime_factorization
Generates prime factorization tasks

Default configuration:
```python
min_value = 2
max_value = 1000
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the prime factorization of 656. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 2 × 2 × 2 × 41
Metadata: {'number': 656, 'factors': [2, 2, 2, 2, 41], 'difficulty': {'value': 656}}

Example 2:
Question: Find the prime factorization of 41. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 41
Metadata: {'number': 41, 'factors': [41], 'difficulty': {'value': 41}}

Example 3:
Question: Find the prime factorization of 420. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)
Answer: 2 × 2 × 3 × 5 × 7
Metadata: {'number': 420, 'factors': [2, 2, 3, 5, 7], 'difficulty': {'value': 420}}

````

### products
Generates multiplication tasks with configurable number of terms

Default configuration:
```python
min_terms = 2
max_terms = 2
min_digits = 1
max_digits = 5
allow_negation = False
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve the following multiplication: 4 * 3. Give only the result as your final answer.
Answer: 12
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 1}, 'expression': '4 * 3'}

Example 2:
Question: Solve the following multiplication: 812 * 880. Give only the result as your final answer.
Answer: 714560
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 3}, 'expression': '812 * 880'}

Example 3:
Question: Solve the following multiplication: 81037 * 25290. Give only the result as your final answer.
Answer: 2049425730
Metadata: {'difficulty': {'num_terms': 2, 'num_digits': 5}, 'expression': '81037 * 25290'}

````

### propositional_logic
Generates propositional logic reasoning tasks

Default configuration:
```python
min_vars = 2
max_vars = 4
min_statements = 2
max_statements = 4
min_complexity = 1
max_complexity = 3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: The following question is a propositional logic reasoning question.

In the question we provide a list of premises. The task is to infer a correct conclusion from the premise.

FORMAT INSTRUCTIONS:
- Return the conclusion logic statement, as your final answer.
- Use the following notation to denote symbols
    - OR = ∨
    - AND = ∧
    - IMPLIES = →
    - IFF = ↔
    - NOT = ¬

Here is the question:
Given:
1. R
.2. Q
.What can we conclude from the above statements?
Answer: None
Metadata: {'premises': ['R', 'Q'], 'variables': ['P', 'Q', 'R', 'S'], 'complexity': 3, 'example_answer': '(P ∨ Q)', 'difficulty': {'vars': 4, 'statements': 2, 'complexity': (1, 3)}}

Example 2:
Question: The following question is a propositional logic reasoning question.

In the question we provide a list of premises. The task is to infer a correct conclusion from the premise.

FORMAT INSTRUCTIONS:
- Return the conclusion logic statement, as your final answer.
- Use the following notation to denote symbols
    - OR = ∨
    - AND = ∧
    - IMPLIES = →
    - IFF = ↔
    - NOT = ¬

Here is the question:
Given:
1. ((Q → P) ∨ (Q → P))
.2. ((Q ↔ Q) → (P → P))
.3. P
.What can we conclude from the above statements?
Answer: None
Metadata: {'premises': ['((Q → P) ∨ (Q → P))', '((Q ↔ Q) → (P → P))', 'P'], 'variables': ['P', 'Q'], 'complexity': 3, 'example_answer': '(Q ∨ P)', 'difficulty': {'vars': 2, 'statements': 3, 'complexity': (1, 3)}}

Example 3:
Question: The following question is a propositional logic reasoning question.

In the question we provide a list of premises. The task is to infer a correct conclusion from the premise.

FORMAT INSTRUCTIONS:
- Return the conclusion logic statement, as your final answer.
- Use the following notation to denote symbols
    - OR = ∨
    - AND = ∧
    - IMPLIES = →
    - IFF = ↔
    - NOT = ¬

Here is the question:
Given:
1. ((Q ∨ P) ∧ ¬P)
.2. P
.3. ((P ∧ R) ∧ ¬R)
.4. ((Q ↔ R) → ¬Q)
.What can we conclude from the above statements?
Answer: None
Metadata: {'premises': ['((Q ∨ P) ∧ ¬P)', 'P', '((P ∧ R) ∧ ¬R)', '((Q ↔ R) → ¬Q)'], 'variables': ['P', 'Q', 'R'], 'complexity': 3, 'example_answer': '(Q ∧ Q)', 'difficulty': {'vars': 3, 'statements': 4, 'complexity': (1, 3)}}

````

### puzzle24
Default configuration:
```python
operators = ('+', '-', '*', '/')
min_value = 1
max_value = 10
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Make 24 using 4, 3, 9, 8. You can only use each number once. You can use the operators +, -, *, /.
Final answer format instructions:
1. Provide your final answer as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.

Answer: 4 + 3 + 9 + 8
Metadata: {'numbers': [4, 3, 9, 8], 'expression': x0 + x1 + x2 + x3}

Example 2:
Question: Make 24 using 8, 2, 10, 4. You can only use each number once. You can use the operators +, -, *, /.
Final answer format instructions:
1. Provide your final answer as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.

Answer: 8 + 2 + 10 + 4
Metadata: {'numbers': [8, 2, 10, 4], 'expression': x0 + x1 + x2 + x3}

Example 3:
Question: Make 24 using 6, 5, 10, 3. You can only use each number once. You can use the operators +, -, *, /.
Final answer format instructions:
1. Provide your final answer as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.

Answer: 6 + 5 + 10 + 3
Metadata: {'numbers': [6, 5, 10, 3], 'expression': x0 + x1 + x2 + x3}

````

### quantum_lock
Generates QuantumLock tasks

Default configuration:
```python
difficulty = 10
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: 0 (red)
Target: 8
Buttons:
A: Add 2 (when any)
B: Add 1 (when green)
C: Multiply 2 (when green)
Answer: A → A → A → A
Metadata: {'metadata': {'difficulty': 2}, 'solution_path': ['A', 'A', 'A', 'A'], 'target_value': 8, 'buttons': [{'name': 'A', 'type': 'add', 'value': 2, 'active_state': 'any'}, {'name': 'B', 'type': 'add', 'value': 1, 'active_state': 'green'}, {'name': 'C', 'type': 'multiply', 'value': 2, 'active_state': 'green'}], 'initial_state': 'red', 'initial_value': 0}

Example 2:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: 0 (red)
Target: 8
Buttons:
A: Subtract 2 (when any)
B: Multiply 2 (when any)
C: Add 2 (when any)
Answer: C → B → B
Metadata: {'metadata': {'difficulty': 1}, 'solution_path': ['C', 'B', 'B'], 'target_value': 8, 'buttons': [{'name': 'A', 'type': 'subtract', 'value': 2, 'active_state': 'any'}, {'name': 'B', 'type': 'multiply', 'value': 2, 'active_state': 'any'}, {'name': 'C', 'type': 'add', 'value': 2, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

Example 3:
Question: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: 0 (red)
Target: 27
Buttons:
A: Multiply 2 (when any)
B: Add 3 (when any)
C: Add 2 (when any)
Answer: B → A → A → A → B
Metadata: {'metadata': {'difficulty': 7}, 'solution_path': ['B', 'A', 'A', 'A', 'B'], 'target_value': 27, 'buttons': [{'name': 'A', 'type': 'multiply', 'value': 2, 'active_state': 'any'}, {'name': 'B', 'type': 'add', 'value': 3, 'active_state': 'any'}, {'name': 'C', 'type': 'add', 'value': 2, 'active_state': 'any'}], 'initial_state': 'red', 'initial_value': 0}

````

### ransom_note
Generates Ransom Note exercises with configurable difficulty

Default configuration:
```python
min_note_length = 1
max_note_length = 10
min_magazine_length = 2
max_magazine_length = 30
p_solvable = 0.5
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: jj
Magazine: nj

Answer: False
Metadata: {'ransom_note': 'jj', 'magazine': 'nj', 'solution': False, 'solvable': False, 'difficulty': {'note_length': 2, 'magazine_length': 2}}

Example 2:
Question: Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: r
Magazine: vipyjtlzxpb

Answer: False
Metadata: {'ransom_note': 'r', 'magazine': 'vipyjtlzxpb', 'solution': False, 'solvable': False, 'difficulty': {'note_length': 1, 'magazine_length': 11}}

Example 3:
Question: Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: dxjijqv
Magazine: vbfvvsgxvsjdxdtmmlhcjci

Answer: False
Metadata: {'ransom_note': 'dxjijqv', 'magazine': 'vbfvvsgxvsjdxdtmmlhcjci', 'solution': False, 'solvable': False, 'difficulty': {'note_length': 7, 'magazine_length': 23}}

````

### rearc
Default configuration:
```python
min_examples = 3
max_examples = 5
diff_lb = 0
diff_ub = 0.2
board_format_opts = BoardFormattingOptions(alphabet=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], col_delimiter=' ', row_delimiter='\n', array_brackets=False)
seed = 42
size = 500
rng_difficulty_ranges = [(0.0, 0.025), (0.025, 0.05), (0.05, 0.075), (0.075, 0.1), (0.1, 0.125), (0.125, 0.15), (0.15, 0.2)]
rng_difficulty_weights = [0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285]
pso_difficulty_ranges = [(0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 1)]
pso_difficulty_weights = [0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285]
```

Example tasks:
````
Example 1:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
1 1 1
1 3 3
1 1 1
Output:
3 3

Example 2:

Input:
4 3 8
4 4 4
Output:
3 8

Example 3:

Input:
3 3 3 3 3 3
3 3 3 5 2 3
3 3 3 5 3 3
3 3 5 3 3 3
3 3 3 3 3 3
3 3 3 3 3 3
Output:
0 5 2
0 5 0
5 0 0


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
4 4 4 4 4
4 4 4 4 4
4 6 8 6 4
4 4 4 4 4
4 4 4 4 4

Answer: 6 8 6
Metadata: {'input': ((4, 4, 4, 4, 4), (4, 4, 4, 4, 4), (4, 6, 8, 6, 4), (4, 4, 4, 4, 4), (4, 4, 4, 4, 4)), 'output': ((6, 8, 6),), 'task_id': 'a740d043', 'difficulty': {'rng': 0.12323282396873297, 'pso': 0.29851851851851846}}

Example 2:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 1 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
2 2 2 2 2 2 2
2 3 2 2 2 2 2
Output:
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
8 8 8 8 8 8 8
2 2 2 2 2 2 2
2 2 2 2 2 2 2

Example 2:

Input:
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 9 7 7 7 7 7 7
7 7 7 7 7 7 7 6
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
Output:
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7

Example 3:

Input:
2 9 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
Output:
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2

Example 4:

Input:
7 7 7 4 7 7 7
7 7 7 7 7 7 7
7 7 7 7 7 7 7
7 7 7 7 7 7 7
Output:
7 7 7 7 7 7 7
7 7 7 7 7 7 7
7 7 7 7 7 7 7
7 7 7 7 7 7 7


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
6 6 6
6 6 6
6 6 6
6 6 6
6 6 6

Answer: 6 6 6
6 6 6
6 6 6
6 6 6
6 6 6
Metadata: {'input': ((6, 6, 6), (6, 6, 6), (6, 6, 6), (6, 6, 6), (6, 6, 6)), 'output': ((6, 6, 6), (6, 6, 6), (6, 6, 6), (6, 6, 6), (6, 6, 6)), 'task_id': 'e26a3af2', 'difficulty': {'rng': 0.11027040425316172, 'pso': 0.061111111111111116}}

Example 3:
Question: Find the common rule that maps an input grid to an output grid, given the examples below.

Example 1:

Input:
0 0 0 2 0 2 2 2 4 0
0 0 0 4 0 0 2 2 4 0
0 0 0 2 4 4 4 4 4 0
4 4 0 2 2 4 4 4 2 2
4 0 4 2 2 0 4 0 2 4
0 2 0 4 4 0 0 2 2 4
2 0 4 2 2 0 4 4 0 2
0 4 2 2 4 2 4 2 0 0
4 0 4 4 0 0 2 2 0 4
Output:
1 1 1 2 0 2 2 2 4 0
1 1 1 4 0 0 2 2 4 0
1 1 1 2 4 4 4 4 4 0
4 4 0 2 2 4 4 4 2 2
4 0 4 2 2 0 4 0 2 4
0 2 0 4 4 0 0 2 2 4
2 0 4 2 2 0 4 4 0 2
0 4 2 2 4 2 4 2 0 0
4 0 4 4 0 0 2 2 0 4

Example 2:

Input:
9 7 0 7 7 9 9
7 7 9 9 0 7 9
0 0 7 0 9 9 0
9 7 7 7 0 0 9
0 0 0 0 7 7 7
0 0 0 9 9 9 0
0 0 0 9 0 9 9
Output:
9 7 0 7 7 9 9
7 7 9 9 0 7 9
0 0 7 0 9 9 0
9 7 7 7 0 0 9
1 1 1 0 7 7 7
1 1 1 9 9 9 0
1 1 1 9 0 9 9

Example 3:

Input:
0 5 5 5 0 0 5 0 5
0 0 5 5 5 5 5 5 5
0 5 5 5 0 0 0 5 5
0 5 5 5 0 0 0 0 5
0 5 0 5 0 0 0 0 5
0 0 5 5 5 5 5 5 0
0 5 5 5 5 5 0 5 5
5 5 5 5 5 0 5 5 0
5 0 5 0 0 5 5 5 5
0 5 5 5 5 5 5 0 0
Output:
0 5 5 5 0 0 5 0 5
0 0 5 5 5 5 5 5 5
0 5 5 5 1 1 1 5 5
0 5 5 5 1 1 1 0 5
0 5 0 5 1 1 1 0 5
0 0 5 5 5 5 5 5 0
0 5 5 5 5 5 0 5 5
5 5 5 5 5 0 5 5 0
5 0 5 0 0 5 5 5 5
0 5 5 5 5 5 5 0 0

Example 4:

Input:
2 2 0 0 2 2 5 2 5
5 5 2 2 5 2 5 2 2
2 0 2 2 2 0 0 0 0
0 5 2 2 0 0 0 0 0
2 0 5 5 0 0 0 0 5
5 5 5 0 2 0 0 5 5
2 5 0 5 5 5 2 5 2
Output:
2 2 0 0 2 2 5 2 5
5 5 2 2 5 2 5 2 2
2 0 2 2 2 1 1 1 0
0 5 2 2 0 1 1 1 0
2 0 5 5 0 1 1 1 5
5 5 5 0 2 0 0 5 5
2 5 0 5 5 5 2 5 2

Example 5:

Input:
2 2 2 0 2 2 2 2
2 0 0 2 0 0 0 0
2 0 2 2 0 0 0 0
2 2 0 0 0 0 0 0
0 2 0 0 0 0 2 2
2 2 0 0 0 2 0 2
0 0 2 2 2 2 2 2
2 2 2 2 2 2 2 0
Output:
2 2 2 0 2 2 2 2
2 0 0 2 1 1 1 1
2 0 2 2 1 1 1 1
2 2 1 1 1 1 1 1
0 2 1 1 1 0 2 2
2 2 1 1 1 2 0 2
0 0 2 2 2 2 2 2
2 2 2 2 2 2 2 0


Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.

Input:
8 8 8 0 0 0 0
8 8 8 0 0 0 8
0 8 8 0 0 0 0
8 8 8 0 8 8 8
0 8 8 8 8 8 8
0 8 8 8 0 8 8

Answer: 8 8 8 1 1 1 0
8 8 8 1 1 1 8
0 8 8 1 1 1 0
8 8 8 0 8 8 8
0 8 8 8 8 8 8
0 8 8 8 0 8 8
Metadata: {'input': ((8, 8, 8, 0, 0, 0, 0), (8, 8, 8, 0, 0, 0, 8), (0, 8, 8, 0, 0, 0, 0), (8, 8, 8, 0, 8, 8, 8), (0, 8, 8, 8, 8, 8, 8), (0, 8, 8, 8, 0, 8, 8)), 'output': ((8, 8, 8, 1, 1, 1, 0), (8, 8, 8, 1, 1, 1, 8), (0, 8, 8, 1, 1, 1, 0), (8, 8, 8, 0, 8, 8, 8), (0, 8, 8, 8, 8, 8, 8), (0, 8, 8, 8, 0, 8, 8)), 'task_id': '6cf79266', 'difficulty': {'rng': 0.04912998774545625, 'pso': 0.17507936507936508}}

````

### rectangle_count
Generates ASCII rectangle counting puzzles with configurable parameters

Default configuration:
```python
max_rectangles = 10
width = 80
height = 80
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to count how many rectangles are present in an ASCII grid.

Single rectangles are outlined with a '#', overlapping rectangles (max 2) are shown with '█'.

Your output should be a single number, representing the total count of rectangles.

Now, it's your turn. How many rectangles do you see in the grid below?
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                 ##################################################             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 #                                                #             
                 ##################################################             
                                                                                
                                                                                
                                                                                
                                                                                
   ######################################                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   #                                    #                                       
   ######################################                                       
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                


Answer: 2
Metadata: {'puzzle': '                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                 ##################################################             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 #                                                #             \n                 ##################################################             \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n   ######################################                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   #                                    #                                       \n   ######################################                                       \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n', 'solution': 2, 'difficulty': {'max_rectangles': 2}}

Example 2:
Question: Your task is to count how many rectangles are present in an ASCII grid.

Single rectangles are outlined with a '#', overlapping rectangles (max 2) are shown with '█'.

Your output should be a single number, representing the total count of rectangles.

Now, it's your turn. How many rectangles do you see in the grid below?
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                    ############                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    #          #                                
                                    ############                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                


Answer: 1
Metadata: {'puzzle': '                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                    ############                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    #          #                                \n                                    ############                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n', 'solution': 1, 'difficulty': {'max_rectangles': 1}}

Example 3:
Question: Your task is to count how many rectangles are present in an ASCII grid.

Single rectangles are outlined with a '#', overlapping rectangles (max 2) are shown with '█'.

Your output should be a single number, representing the total count of rectangles.

Now, it's your turn. How many rectangles do you see in the grid below?
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                         #########################              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       ############   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                    #####█#######################██#########█#  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #    #                       ##         ##  
                                    #####█#######################██#########█#  
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #                       ##         #   
                                         #      ##########       ##         #   
                                         #      #        #       ############   
                                         #      #        #       #              
                                         #      ##########       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #                       #              
                                         #########################              
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
            #######################                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #                     #                                             
            #               ######█###                                          
            #               #     #  #                                          
            #               ######█###                                          
            #                     #   ###########################               
            #                     #   #                         #               
            #                     #   #                         #               
            #######################   ###########################               
                                                                                


Answer: 7
Metadata: {'puzzle': '                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                         #########################              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       ############   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                    #####█#######################██#########█#  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #    #                       ##         ##  \n                                    #####█#######################██#########█#  \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #                       ##         #   \n                                         #      ##########       ##         #   \n                                         #      #        #       ############   \n                                         #      #        #       #              \n                                         #      ##########       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #                       #              \n                                         #########################              \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n                                                                                \n            #######################                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #                     #                                             \n            #               ######█###                                          \n            #               #     #  #                                          \n            #               ######█###                                          \n            #                     #   ###########################               \n            #                     #   #                         #               \n            #                     #   #                         #               \n            #######################   ###########################               \n                                                                                \n', 'solution': 7, 'difficulty': {'max_rectangles': 7}}

````

### rotate_matrix
Generates Rotate Matrix exercises with configurable difficulty

Default configuration:
```python
min_n = 2
max_n = 10
min_rotations = 0
max_rotations = 10
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a square matrix, your job is to rotate it clockwise.

Your output should be a matrix in the same format as the input.

Rotate the matrix below by 540 degrees clockwise:
0 4 3
3 2 1
8 1 9

Answer: 9 1 8
1 2 3
3 4 0
Metadata: {'matrix': [[0, 4, 3], [3, 2, 1], [8, 1, 9]], 'num_rotations': 6, 'solution': [[9, 1, 8], [1, 2, 3], [3, 4, 0]], 'difficulty': {'n': 3, 'num_rotations': 6}}

Example 2:
Question: Given a square matrix, your job is to rotate it clockwise.

Your output should be a matrix in the same format as the input.

Rotate the matrix below by 900 degrees clockwise:
4 2
7 5

Answer: 5 7
2 4
Metadata: {'matrix': [[4, 2], [7, 5]], 'num_rotations': 10, 'solution': [[5, 7], [2, 4]], 'difficulty': {'n': 2, 'num_rotations': 10}}

Example 3:
Question: Given a square matrix, your job is to rotate it clockwise.

Your output should be a matrix in the same format as the input.

Rotate the matrix below by 180 degrees clockwise:
8 8 1 2 6 3 4 0
3 1 9 0 1 2 8 4
6 9 6 5 5 1 5 4
9 2 1 8 1 9 1 4
5 1 4 0 5 6 1 7
7 3 3 2 4 3 0 0
6 0 5 5 7 7 9 8
2 3 7 7 5 9 0 4

Answer: 4 0 9 5 7 7 3 2
8 9 7 7 5 5 0 6
0 0 3 4 2 3 3 7
7 1 6 5 0 4 1 5
4 1 9 1 8 1 2 9
4 5 1 5 5 6 9 6
4 8 2 1 0 9 1 3
0 4 3 6 2 1 8 8
Metadata: {'matrix': [[8, 8, 1, 2, 6, 3, 4, 0], [3, 1, 9, 0, 1, 2, 8, 4], [6, 9, 6, 5, 5, 1, 5, 4], [9, 2, 1, 8, 1, 9, 1, 4], [5, 1, 4, 0, 5, 6, 1, 7], [7, 3, 3, 2, 4, 3, 0, 0], [6, 0, 5, 5, 7, 7, 9, 8], [2, 3, 7, 7, 5, 9, 0, 4]], 'num_rotations': 2, 'solution': [[4, 0, 9, 5, 7, 7, 3, 2], [8, 9, 7, 7, 5, 5, 0, 6], [0, 0, 3, 4, 2, 3, 3, 7], [7, 1, 6, 5, 0, 4, 1, 5], [4, 1, 9, 1, 8, 1, 2, 9], [4, 5, 1, 5, 5, 6, 9, 6], [4, 8, 2, 1, 0, 9, 1, 3], [0, 4, 3, 6, 2, 1, 8, 8]], 'difficulty': {'n': 8, 'num_rotations': 2}}

````

### rotten_oranges
Generates Rotten Oranges exercises with configurable difficulty

Default configuration:
```python
min_n = 10
max_n = 30
p_oranges = 0.85
p_rotten = 0.1
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: You are given an n x n grid where each cell can have one of three values:
- 0 representing an empty cell
- 1 representing a fresh orange
- 2 representing a rotten orange

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Your task is determine the minimum number of minutes that must elapse until no cell has a fresh orange.
If this is impossible, return -1.

Now, determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
1 1 1 1 2 1 1 1 1 0 1 1 1 1 1 2 1 0 1 1 1 1 1 0 0 1 1 1 1 1
1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 2 1 0
1 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 2 1 1 2 1 1
1 1 1 1 1 1 0 1 2 1 1 1 1 0 1 0 1 2 1 1 1 0 2 1 1 1 1 1 2 2
2 1 2 1 2 0 1 1 2 1 1 1 1 1 0 0 1 2 1 1 1 1 1 0 1 1 0 1 1 1
1 1 0 1 0 1 2 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1
1 1 1 1 1 0 1 1 1 1 1 2 0 1 0 1 1 1 1 2 1 1 0 1 1 0 1 1 1 1
1 1 1 1 1 1 1 1 1 0 1 1 2 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 1
1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 1 1 0 1 1
1 1 0 0 1 1 1 0 1 1 1 1 1 1 1 2 0 2 1 1 1 0 1 1 0 1 1 1 1 1
1 1 1 1 1 1 2 2 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 2 1 1 1 2 1 1 1 1
1 1 1 0 1 1 1 1 1 1 1 1 2 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1
1 1 2 1 1 1 1 0 1 0 1 1 1 1 1 2 1 1 2 0 2 1 1 1 1 0 2 1 1 1
1 1 1 0 1 1 1 1 1 2 1 1 2 1 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1
2 0 1 0 0 1 1 2 1 1 1 1 1 1 2 0 1 1 2 2 1 1 1 1 1 1 1 1 0 1
2 0 0 1 1 1 0 1 1 2 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1
0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 1 1 1 1 1 0 1 1
1 1 1 0 1 2 1 0 2 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 2 1 1 1
1 1 2 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 1 1 1 0 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 2 1 1 1 1 1 1
1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 1 2 1 1 1 1 1 1 2
0 1 1 1 1 1 1 1 1 2 2 1 1 1 1 0 2 0 1 1 0 1 1 1 1 0 1 1 1 2
1 1 1 0 0 1 1 0 1 1 2 1 1 1 0 0 1 2 1 1 1 1 1 1 1 0 1 1 1 0
2 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 0 1 1 1 2 1 2 0 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 1 1 2 1 1 1 1 1 1 1 1
0 1 1 1 1 2 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 0 1 1 2 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0
1 1 1 1 1 2 1 1 1 1 1 1 1 0 1 1 1 0 1 2 1 1 1 1 1 1 1 1 2 0
1 1 1 1 1 1 1 1 1 1 1 1 2 0 0 1 0 1 1 1 1 2 1 1 1 1 1 1 1 1

Answer: 6
Metadata: {'matrix': [[1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 1], [1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1, 2, 2], [2, 1, 2, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 1, 2, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], [2, 0, 1, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], [2, 0, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 1, 2, 1, 0, 2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1], [1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 2], [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2], [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0], [2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 2, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]], 'solution': 6, 'difficulty': {'n': 30}}

Example 2:
Question: You are given an n x n grid where each cell can have one of three values:
- 0 representing an empty cell
- 1 representing a fresh orange
- 2 representing a rotten orange

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Your task is determine the minimum number of minutes that must elapse until no cell has a fresh orange.
If this is impossible, return -1.

Now, determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
1 0 1 1 1 1 0 0 0 2 1
1 1 1 1 1 2 1 1 0 1 2
1 1 1 1 1 0 1 2 0 1 0
1 1 1 1 0 1 1 1 1 1 2
1 1 1 1 1 2 1 1 0 1 1
2 1 1 1 1 1 1 1 2 0 1
1 1 1 1 1 1 1 1 1 1 1
1 0 1 1 2 1 1 1 0 1 1
1 1 1 1 1 1 2 1 1 1 1
0 2 1 1 1 1 0 1 1 1 1
1 0 1 1 1 1 1 1 0 1 1

Answer: -1
Metadata: {'matrix': [[1, 0, 1, 1, 1, 1, 0, 0, 0, 2, 1], [1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 2], [1, 1, 1, 1, 1, 0, 1, 2, 0, 1, 0], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1], [0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]], 'solution': -1, 'difficulty': {'n': 11}}

Example 3:
Question: You are given an n x n grid where each cell can have one of three values:
- 0 representing an empty cell
- 1 representing a fresh orange
- 2 representing a rotten orange

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Your task is determine the minimum number of minutes that must elapse until no cell has a fresh orange.
If this is impossible, return -1.

Now, determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 2 0 1 1 1 1 1
1 0 1 1 1 1 1 1 0 1 1 1 0 1 1 0 2 1 2 1 1 0 0
1 0 0 0 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1 1 1 0 1
0 0 2 0 1 1 1 0 1 1 0 2 1 1 2 2 0 1 1 2 1 0 1
1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 1 1
1 1 2 1 1 1 1 2 1 1 1 1 1 1 2 2 1 1 1 1 1 0 1
1 1 1 1 1 2 1 1 2 1 1 1 1 0 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 2 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1
0 1 1 1 0 0 1 1 1 0 1 1 0 2 1 1 2 1 0 1 2 0 1
1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 2 1 1 0 1 1 1 1
1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1
1 1 1 0 0 1 1 0 1 1 1 1 2 1 1 0 1 0 1 1 1 1 1
2 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1
1 1 1 0 0 0 1 2 1 1 1 1 1 2 0 1 1 1 1 1 1 1 0
1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 0 2 1 1 1 1 2
1 1 1 1 1 1 1 1 1 2 0 1 1 0 2 1 1 1 1 1 1 1 1
1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1
2 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 2 1
1 1 0 1 2 1 1 1 1 1 2 1 1 2 1 1 1 1 1 1 1 1 0
1 1 0 1 1 1 1 2 1 1 1 2 1 1 1 0 1 2 1 1 1 1 1
1 2 1 1 2 1 0 1 0 1 2 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 0 1 2 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1
1 1 0 1 1 2 1 1 1 1 1 0 1 1 0 1 1 1 0 1 0 0 1

Answer: 13
Metadata: {'matrix': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 1, 2, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1], [0, 0, 2, 0, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 2, 2, 0, 1, 1, 2, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 2, 1, 0, 1, 2, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 1], [1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1], [1, 2, 1, 1, 2, 1, 0, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1]], 'solution': 13, 'difficulty': {'n': 23}}

````

### rubiks_cube
Generates RubiksCube tasks

Default configuration:
```python
min_scramble_steps = 3
max_scramble_steps = 10
cube_size = 3
remove_ansi = True
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You see a size 3 Rubik's cube. It is arranged this:

          G  Y  W                   
          G  Y  W                   
          R  R  R                   
 R  R  B  W  W  W  G  O  O  G  B  Y 
 R  R  W  G  G  G  Y  O  O  G  B  Y 
 R  R  Y  B  B  B  W  O  O  G  B  Y 
          O  O  O                   
          B  W  Y                   
          B  W  Y                   
 

Please provide a solution to solve this cube using Singmaster notation. Do not combine any steps, for instance, do not write 'U2', and instead write 'U U'.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 4, 'scramble_moves': "L' R R F", 'example_correct_answer': "U L D F' R' D' D' F' D B U' B' R' U U R U' L' U L U F U' F' U' L' U L U F U' F' U L' U L U F U' F' U F' U F U R U' R' U F R U R' U' R U R' U' F' U U R U R' U R U U R' U' R U R' U R U U R' U' R U' L' U R' U' L U R U' L' U R' U' L U R' D' R D R' D' R D R' D' R D R' D' R D U R' D' R D R' D' R D U U", 'difficulty': {'scramble_steps': 4, 'cube_size': 3}}

Example 2:
Question: You see a size 3 Rubik's cube. It is arranged this:

          Y  Y  Y                   
          Y  Y  Y                   
          B  R  B                   
 B  B  W  R  G  R  Y  G  G  O  O  O 
 R  R  W  R  G  R  Y  O  O  B  B  B 
 B  B  W  R  G  R  Y  G  G  O  O  O 
          G  O  G                   
          W  W  W                   
          W  W  W                   
 

Please provide a solution to solve this cube using Singmaster notation. Do not combine any steps, for instance, do not write 'U2', and instead write 'U U'.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 3, 'scramble_moves': "D U' F", 'example_correct_answer': "L D L' D' L D' B D R D R' D' R D' F D B U B' U R U' R' U' B U' B' R' U U R U U L U' L' B' U U B F U' F' L' U' L U F U' F' U' R U' R' U' F' U F U B U' B' U' R' U R U' B' U B U L U' L' U' B' U B U L U' L' U B' U B U L U' L' U F R U R' U' F' U R U R' U R U U R' U' R U R' U R U U R' U' L U' R' U L' U' D' R D R' D' R D R' D' R D R' D' R D U U R' D' R D R' D' R D U U", 'difficulty': {'scramble_steps': 3, 'cube_size': 3}}

Example 3:
Question: You see a size 3 Rubik's cube. It is arranged this:

          Y  R  B                   
          G  Y  W                   
          W  W  W                   
 O  R  R  B  B  B  O  O  Y  O  B  G 
 O  R  Y  G  G  G  W  O  R  Y  B  G 
 O  Y  G  W  W  R  B  O  R  Y  B  G 
          R  R  Y                   
          B  W  Y                   
          W  O  G                   
 

Please provide a solution to solve this cube using Singmaster notation. Do not combine any steps, for instance, do not write 'U2', and instead write 'U U'.
Answer: None
Metadata: {'cube_size': 3, 'scramble_steps': 9, 'scramble_moves': "B R' R' L L L B' F' B'", 'example_correct_answer': "U U B B F L D D L D D U R U' R' F' U U F U' R' U R U U L U' L' B' U U B U' U' L' U L U F U' F' U R U' R' U' F' U F R' U R U B U' B' U' B' U B U L U' L' U' F R U R' U' R U R' U' F' U R U' L' U R' U' L U L U' R' U L' U' R U L U' R' U L' U' R", 'difficulty': {'scramble_steps': 9, 'cube_size': 3}}

````

### rush_hour
Generates Rush Hour puzzle configurations from pre-computed database

Default configuration:
```python
min_moves = 1
max_moves = 50
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Move the red car (AA) to the exit on the right.
Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'
where the letter is the vehicle and +/- number is spaces to move right/left or down/up.
Walls are marked with an 'x'. Cars cannot move through walls, and walls cannot be moved.
A car oriented vertically can only move up and down, a car oriented horizontally can only move left and right.

Board:
.xBBCC
..x.K.
G.AAK.
G.IJDD
H.IJ..
HEEFFF

Answer: None
Metadata: {'board_config': 'oxCCDDooxoMoIoAAMoIoKLFFJoKLooJGGHHH', 'min_moves': 10, 'difficulty': {'min_moves': 10}}

Example 2:
Question: Move the red car (AA) to the exit on the right.
Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'
where the letter is the vehicle and +/- number is spaces to move right/left or down/up.
Walls are marked with an 'x'. Cars cannot move through walls, and walls cannot be moved.
A car oriented vertically can only move up and down, a car oriented horizontally can only move left and right.

Board:
EBBCCC
E....H
F.xAAH
F.G...
..GDDD
......

Answer: None
Metadata: {'board_config': 'FCCDDDFooooIGoxAAIGoHoooooHEEEoooooo', 'min_moves': 6, 'difficulty': {'min_moves': 6}}

Example 3:
Question: Move the red car (AA) to the exit on the right.
Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'
where the letter is the vehicle and +/- number is spaces to move right/left or down/up.
Walls are marked with an 'x'. Cars cannot move through walls, and walls cannot be moved.
A car oriented vertically can only move up and down, a car oriented horizontally can only move left and right.

Board:
GBBIJK
G..IJK
AAHI..
..HCCC
..xDD.
EEEFF.

Answer: None
Metadata: {'board_config': 'HBBJKLHooJKLAAIJooooICCCooxEEoFFFGGo', 'min_moves': 30, 'difficulty': {'min_moves': 30}}

````

### self_reference
Generates self-referential puzzles

Default configuration:
```python
difficulty = 5
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Given the truthfulness of these statements, please tell me the number of possible solutions: 
 - Statement 1: 'At least 1 of these 7 statements are true.'
 - Statement 2: 'At most 3 of these 7 statements are false.'
 - Statement 3: 'Exactly 4 of these 7 statements are true.'
 - Statement 4: 'Exactly 3 of these 7 statements are false.'
 - Statement 5: 'Either Statement 3 or Statement 4 is true, but not both.'
 - Statement 6: 'The number of true statements is a prime number.'
 - Statement 7: 'The number of false statements is a composite number.'

Answer: 4
Metadata: {'difficulty': 5}

Example 2:
Question: Given the truthfulness of these statements, please tell me the number of possible solutions: 
 - Statement 1: 'At least 4 of these 7 statements are true.'
 - Statement 2: 'At most 5 of these 7 statements are false.'
 - Statement 3: 'Exactly 7 of these 7 statements are true.'
 - Statement 4: 'Exactly 1 of these 7 statements are false.'
 - Statement 5: 'Either Statement 3 or Statement 4 is true, but not both.'
 - Statement 6: 'The number of true statements is a prime number.'
 - Statement 7: 'The number of false statements is a composite number.'

Answer: 4
Metadata: {'difficulty': 5}

Example 3:
Question: Given the truthfulness of these statements, please tell me the number of possible solutions: 
 - Statement 1: 'At least 2 of these 7 statements are true.'
 - Statement 2: 'At most 5 of these 7 statements are false.'
 - Statement 3: 'Exactly 0 of these 7 statements are true.'
 - Statement 4: 'Exactly 3 of these 7 statements are false.'
 - Statement 5: 'Either Statement 3 or Statement 4 is true, but not both.'
 - Statement 6: 'The number of true statements is a prime number.'
 - Statement 7: 'The number of false statements is a composite number.'

Answer: 2
Metadata: {'difficulty': 5}

````

### sentence_reordering
Generates sentence reordering tasks from text spans

Default configuration:
```python
min_words_in_sentence = 3
max_words_in_sentence = 20
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Restore the correct order of words in the following sentence: wish could get I sleep. "I some
Answer: "I wish I could get some sleep.
Metadata: {'word_count': 7, 'difficulty': {'words_in_sentence': 7}}

Example 2:
Question: Restore the correct order of words in the following sentence: the high level name. itself its unable it maintain at was of to Unfortunately,
Answer: Unfortunately, it was unable to maintain itself at the high level of its name.
Metadata: {'word_count': 14, 'difficulty': {'words_in_sentence': 14}}

Example 3:
Question: Restore the correct order of words in the following sentence: developed by For the unutilized. energy falls ages went the
Answer: For ages the the energy developed by falls went unutilized.
Metadata: {'word_count': 10, 'difficulty': {'words_in_sentence': 10}}

````

### shortest_path
Generates Shortest Path exercises with configurable difficulty

Default configuration:
```python
min_rows = 5
max_rows = 8
min_cols = 5
max_cols = 8
p_blocked = 0.4
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your task is to find the shortest path from the start to the destination point in a grid.

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the shortest path from * to #, moving only through open cells.

You may only move in four directions: up, down, left, and right.

If there is no path from * to #, simply write "infeasible" (without quotes).

Your output should be a sequence of directions that leads from * to #, e.g. right right down down up left

Now, find the length of the shortest path from * to # in the following grid:
O X X X O
O O X X X
O O # O O
* X O O X
O X X O X

Answer: up right right
Metadata: {'matrix': [['O', 'X', 'X', 'X', 'O'], ['O', 'O', 'X', 'X', 'X'], ['O', 'O', '#', 'O', 'O'], ['*', 'X', 'O', 'O', 'X'], ['O', 'X', 'X', 'O', 'X']], 'solution': ['up', 'right', 'right'], 'difficulty': {'rows': 5, 'cols': 5}}

Example 2:
Question: Your task is to find the shortest path from the start to the destination point in a grid.

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the shortest path from * to #, moving only through open cells.

You may only move in four directions: up, down, left, and right.

If there is no path from * to #, simply write "infeasible" (without quotes).

Your output should be a sequence of directions that leads from * to #, e.g. right right down down up left

Now, find the length of the shortest path from * to # in the following grid:
# X O O O O O
X O X O O O O
X O O X X O O
O O O O X X X
O O X O O * O

Answer: infeasible
Metadata: {'matrix': [['#', 'X', 'O', 'O', 'O', 'O', 'O'], ['X', 'O', 'X', 'O', 'O', 'O', 'O'], ['X', 'O', 'O', 'X', 'X', 'O', 'O'], ['O', 'O', 'O', 'O', 'X', 'X', 'X'], ['O', 'O', 'X', 'O', 'O', '*', 'O']], 'solution': [], 'difficulty': {'rows': 5, 'cols': 7}}

Example 3:
Question: Your task is to find the shortest path from the start to the destination point in a grid.

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the shortest path from * to #, moving only through open cells.

You may only move in four directions: up, down, left, and right.

If there is no path from * to #, simply write "infeasible" (without quotes).

Your output should be a sequence of directions that leads from * to #, e.g. right right down down up left

Now, find the length of the shortest path from * to # in the following grid:
X X X X X
X O O O X
O # X X O
O X X X O
X O O X X
O O X X X
X O O O X
O O O X *

Answer: infeasible
Metadata: {'matrix': [['X', 'X', 'X', 'X', 'X'], ['X', 'O', 'O', 'O', 'X'], ['O', '#', 'X', 'X', 'O'], ['O', 'X', 'X', 'X', 'O'], ['X', 'O', 'O', 'X', 'X'], ['O', 'O', 'X', 'X', 'X'], ['X', 'O', 'O', 'O', 'X'], ['O', 'O', 'O', 'X', '*']], 'solution': [], 'difficulty': {'rows': 8, 'cols': 5}}

````

### simple_equations
Generates simple equations with one variable to solve

Default configuration:
```python
min_terms = 2
max_terms = 4
min_value = 1
max_value = 100
operators = ('+', '-', '*')
operators_weights = [0.4, 0.4, 0.2]
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the value of u in the equation: 32*u + 4 = 3044
Answer: 95
Metadata: {'equation': '32*u + 4 = 3044', 'variable': 'u', 'difficulty': {'min_terms': 2, 'max_terms': 4, 'min_value': 1, 'max_value': 100, 'operators_weights': [0.4, 0.4, 0.2]}}

Example 2:
Question: Determine the value of b that satisfies: 71 - 48*b = -2761
Answer: 59
Metadata: {'equation': '71 - 48*b = -2761', 'variable': 'b', 'difficulty': {'min_terms': 2, 'max_terms': 4, 'min_value': 1, 'max_value': 100, 'operators_weights': [0.4, 0.4, 0.2]}}

Example 3:
Question: Find the value of n in the equation: 175 - 29*n = -202
Answer: 13
Metadata: {'equation': '175 - 29*n = -202', 'variable': 'n', 'difficulty': {'min_terms': 2, 'max_terms': 4, 'min_value': 1, 'max_value': 100, 'operators_weights': [0.4, 0.4, 0.2]}}

````

### simple_geometry
A dataset for simple polygon angle-finding tasks.
    We randomly choose the number of sides N within [min_sides, max_sides].
    We then generate (N-1) random angles (in degrees), ensuring their sum is
    strictly less than the total sum for an (N)-sided convex polygon (which is 180*(N-2)).
    The question asks for the missing angle; the answer is computed by subtracting the
    sum of known angles from 180*(N-2).

Default configuration:
```python
min_sides = 3
max_sides = 6
min_angle = 10
max_angle = 170
seed = 42
size = 100
```

Example tasks:
````
Example 1:
Question: Given a convex polygon with 3 sides, its first 2 interior angles are: 16.0°, 80.0°. What is the measure of the remaining interior angle (in degrees)?Return only the angle as your answer.Do not give the units in your answer.
Answer: 84
Metadata: {'n_sides': 3, 'known_angles': [16.0, 80.0], 'sum_of_known_angles': 96.0, 'missing_angle_raw': 84.0, 'missing_angle_rounded': 84, 'total_interior_sum': 180, 'difficulty': {'sides': 3}}

Example 2:
Question: A convex polygon has 3 sides. The measures of the first 2 interior angles are: 83.0°, 46.0°. Find the measure of the last interior angle.Return only the angle as your answer.Do not give the units in your answer.
Answer: 51
Metadata: {'n_sides': 3, 'known_angles': [83.0, 46.0], 'sum_of_known_angles': 129.0, 'missing_angle_raw': 51.0, 'missing_angle_rounded': 51, 'total_interior_sum': 180, 'difficulty': {'sides': 3}}

Example 3:
Question: Given a convex polygon with 6 sides, its first 5 interior angles are: 143.0°, 148.0°, 39.0°, 55.0°, 107.0°. What is the measure of the remaining interior angle (in degrees)?Return only the angle as your answer.Do not give the units in your answer.
Answer: 228
Metadata: {'n_sides': 6, 'known_angles': [143.0, 148.0, 39.0, 55.0, 107.0], 'sum_of_known_angles': 492.0, 'missing_angle_raw': 228.0, 'missing_angle_rounded': 228, 'total_interior_sum': 720, 'difficulty': {'sides': 6}}

````

### simple_integration
Generates simple integration problems with one variable

Default configuration:
```python
min_terms = 2
max_terms = 5
min_degree = 1
max_degree = 10
min_bounds = 1
max_bounds = 10
operators = ('+', '-')
symbols = ('x', 'X')
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Find the indefinite integral: ∫ 70*x**6 + 12*x**2/5 dx
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.

Answer: 10*x**7 + 4*x**3/5 + C
Metadata: {'integrand': '70*x**6 + 12*x**2/5', 'variable': 'x', 'expected_answer_expression': 10*x**7 + 4*x**3/5, 'difficulty': {'terms': 2}}

Example 2:
Question: Evaluate the indefinite integral: ∫ 48*X**5 - 10/9 dx
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.

Answer: 8*X**6 - 10*X/9 + C
Metadata: {'integrand': '48*X**5 - 10/9', 'variable': 'X', 'expected_answer_expression': 8*X**6 - 10*X/9, 'difficulty': {'terms': 2}}

Example 3:
Question: Evaluate the indefinite integral: ∫ -27*x**8 + 9*x**5/2 - 28*x**3 + 5*x**2 + 8*x dx
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.

Answer: -3*x**9 + 3*x**6/4 - 7*x**4 + 5*x**3/3 + 4*x**2 + C
Metadata: {'integrand': '-27*x**8 + 9*x**5/2 - 28*x**3 + 5*x**2 + 8*x', 'variable': 'x', 'expected_answer_expression': -3*x**9 + 3*x**6/4 - 7*x**4 + 5*x**3/3 + 4*x**2, 'difficulty': {'terms': 5}}

````

### sokoban
Generates Sokoban games with configurable parameters

Default configuration:
```python
min_w = 6
min_h = 6
max_w = 10
max_h = 10
min_boxes = 4
max_boxes = 10
max_depth = 80
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: You are going to solve a 'sokoban' puzzle.

* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box on a goal
+ - A wall
- - An empty position

Your solution must be a string of characters, ex: LDURRUDL.

Here is your puzzle:
+ + + + + +  
+ X X @ - +  
+ - X - - +  
+ @ - @ - +  
+ % @ - - +  
+ + + + + +  


Answer: UURRRULDRDLDLU
Metadata: {'gamestr': '+ + + + + +  \n+ X X @ - +  \n+ - X - - +  \n+ @ - @ - +  \n+ % @ - - +  \n+ + + + + +  \n\n', 'difficulty': {'width': 6, 'height': 6}}

Example 2:
Question: You are going to solve a 'sokoban' puzzle.

* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box on a goal
+ - A wall
- - An empty position

Your solution must be a string of characters, ex: LDURRUDL.

Here is your puzzle:
+ + + + + +  
+ - - - * +  
+ @ @ @ @ +  
+ X - - - +  
+ - - - X +  
+ X - $ X +  
+ + - - - +  
+ + + + + +  


Answer: DDUULDDRDLLULDURRUULDULDRRDDLURUULLDRRURDD
Metadata: {'gamestr': '+ + + + + +  \n+ - - - * +  \n+ @ @ @ @ +  \n+ X - - - +  \n+ - - - X +  \n+ X - $ X +  \n+ + - - - +  \n+ + + + + +  \n\n', 'difficulty': {'width': 6, 'height': 8}}

Example 3:
Question: You are going to solve a 'sokoban' puzzle.

* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box on a goal
+ - A wall
- - An empty position

Your solution must be a string of characters, ex: LDURRUDL.

Here is your puzzle:
+ + + + + + + + + +  
+ $ + + + X - @ - +  
+ + + + + X @ @ - +  
+ X + + X - - X - +  
+ - + X @ - - X - +  
+ @ - - X @ - - - +  
+ - - - @ - @ - - +  
+ - - - - - - * - +  
+ + + + + + + + + +  


Answer: ULULULRDRDDLUUUURRRUULLDDUURDLDLDRLDDDLLULLUUDRRDRDRUUURUL
Metadata: {'gamestr': '+ + + + + + + + + +  \n+ $ + + + X - @ - +  \n+ + + + + X @ @ - +  \n+ X + + X - - X - +  \n+ - + X @ - - X - +  \n+ @ - - X @ - - - +  \n+ - - - @ - @ - - +  \n+ - - - - - - * - +  \n+ + + + + + + + + +  \n\n', 'difficulty': {'width': 10, 'height': 9}}

````

### spell_backward
Generates tasks to spell words backward

Default configuration:
```python
min_word_len = 3
max_word_len = 20
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Spell this word backward (example: sun -> nus): Project
Answer: tcejorP
Metadata: {'word': 'Project', 'word_len': 7, 'difficulty': {'word_len': (3, 20)}}

Example 2:
Question: Spell this word backward (example: sun -> nus): Would
Answer: dluoW
Metadata: {'word': 'Would', 'word_len': 5, 'difficulty': {'word_len': (3, 20)}}

Example 3:
Question: Spell this word backward (example: sun -> nus): One
Answer: enO
Metadata: {'word': 'One', 'word_len': 3, 'difficulty': {'word_len': (3, 20)}}

````

### spiral_matrix
Generates Spiral Matrix exercises with configurable difficulty

Default configuration:
```python
min_n = 2
max_n = 10
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

The spiral order is clockwise, starting from the top-left corner. More precisely:
- Start from the top-left corner and move right.
- Move down towards the bottom-right corner.
- Move left towards the bottom-left corner.
- Move up towards the top-right corner.
- Repeat the steps for the inner elements of the matrix until every entry is visited.

Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6

For the matrix below, what is the list of elements in spiral order?
3 1 3
2 4 9
1 0 8

Answer: 3 1 3 9 8 0 1 2 4
Metadata: {'matrix': [[3, 1, 3], [2, 4, 9], [1, 0, 8]], 'solution': [3, 1, 3, 9, 8, 0, 1, 2, 4], 'difficulty': {'n': 3}}

Example 2:
Question: Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

The spiral order is clockwise, starting from the top-left corner. More precisely:
- Start from the top-left corner and move right.
- Move down towards the bottom-right corner.
- Move left towards the bottom-left corner.
- Move up towards the top-right corner.
- Repeat the steps for the inner elements of the matrix until every entry is visited.

Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6

For the matrix below, what is the list of elements in spiral order?
5 7
2 4

Answer: 5 7 4 2
Metadata: {'matrix': [[5, 7], [2, 4]], 'solution': [5, 7, 4, 2], 'difficulty': {'n': 2}}

Example 3:
Question: Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

The spiral order is clockwise, starting from the top-left corner. More precisely:
- Start from the top-left corner and move right.
- Move down towards the bottom-right corner.
- Move left towards the bottom-left corner.
- Move up towards the top-right corner.
- Repeat the steps for the inner elements of the matrix until every entry is visited.

Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6

For the matrix below, what is the list of elements in spiral order?
1 9 9 5 2 9 7 3
1 1 5 0 7 0 4 9
3 5 4 7 8 4 3 4
6 5 3 3 2 7 1 9
6 7 7 0 1 4 1 8
8 2 5 9 0 1 4 0
2 1 5 5 6 4 0 3
1 6 6 0 2 8 8 5

Answer: 1 9 9 5 2 9 7 3 9 4 9 8 0 3 5 8 8 2 0 6 6 1 2 8 6 6 3 1 1 5 0 7 0 4 3 1 1 4 0 4 6 5 5 1 2 7 5 5 4 7 8 4 7 4 1 0 9 5 7 3 3 2 1 0
Metadata: {'matrix': [[1, 9, 9, 5, 2, 9, 7, 3], [1, 1, 5, 0, 7, 0, 4, 9], [3, 5, 4, 7, 8, 4, 3, 4], [6, 5, 3, 3, 2, 7, 1, 9], [6, 7, 7, 0, 1, 4, 1, 8], [8, 2, 5, 9, 0, 1, 4, 0], [2, 1, 5, 5, 6, 4, 0, 3], [1, 6, 6, 0, 2, 8, 8, 5]], 'solution': [1, 9, 9, 5, 2, 9, 7, 3, 9, 4, 9, 8, 0, 3, 5, 8, 8, 2, 0, 6, 6, 1, 2, 8, 6, 6, 3, 1, 1, 5, 0, 7, 0, 4, 3, 1, 1, 4, 0, 4, 6, 5, 5, 1, 2, 7, 5, 5, 4, 7, 8, 4, 7, 4, 1, 0, 9, 5, 7, 3, 3, 2, 1, 0], 'difficulty': {'n': 8}}

````

### string_insertion
Generates String Insertion exercises with configurable difficulty

Default configuration:
```python
min_string_length = 5
max_string_length = 20
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern:
1. If there is a substring ABCD in the string, insert the character A after the substring.
2. If there is a substring BCDE in the string, insert the character B after the substring.
3. If there is a substring CDEA in the string, insert the character C after the substring.
4. If there is a substring DEAB in the string, insert the character D after the substring.
5. If there is a substring EABC in the string, insert the character E after the substring.

Once you have inserted a character, you have to skip over the substring and the inserted character and continue the search from the next character.

Your output should be a string that has been modified according to the pattern.

Given the following string, provide the answer after inserting the characters according to the pattern: ACBBBAEA

Answer: ACBBBAEA
Metadata: {'string': 'ACBBBAEA', 'solution': 'ACBBBAEA', 'difficulty': {'string_length': 8}}

Example 2:
Question: Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern:
1. If there is a substring ABCD in the string, insert the character A after the substring.
2. If there is a substring BCDE in the string, insert the character B after the substring.
3. If there is a substring CDEA in the string, insert the character C after the substring.
4. If there is a substring DEAB in the string, insert the character D after the substring.
5. If there is a substring EABC in the string, insert the character E after the substring.

Once you have inserted a character, you have to skip over the substring and the inserted character and continue the search from the next character.

Your output should be a string that has been modified according to the pattern.

Given the following string, provide the answer after inserting the characters according to the pattern: CBDCAD

Answer: CBDCAD
Metadata: {'string': 'CBDCAD', 'solution': 'CBDCAD', 'difficulty': {'string_length': 6}}

Example 3:
Question: Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern:
1. If there is a substring ABCD in the string, insert the character A after the substring.
2. If there is a substring BCDE in the string, insert the character B after the substring.
3. If there is a substring CDEA in the string, insert the character C after the substring.
4. If there is a substring DEAB in the string, insert the character D after the substring.
5. If there is a substring EABC in the string, insert the character E after the substring.

Once you have inserted a character, you have to skip over the substring and the inserted character and continue the search from the next character.

Your output should be a string that has been modified according to the pattern.

Given the following string, provide the answer after inserting the characters according to the pattern: EEABDBCABAEAABECDE

Answer: EEABDBCABAEAABECDE
Metadata: {'string': 'EEABDBCABAEAABECDE', 'solution': 'EEABDBCABAEAABECDE', 'difficulty': {'string_length': 18}}

````

### string_manipulation
Generates String Insertion exercises with configurable difficulty

Default configuration:
```python
min_string_length = 5
max_string_length = 20
min_num_rules = 3
max_num_rules = 8
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
1. If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.
2. If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.
3. If the string ends with 'ca', remove the last character.
4. If the string suffix is 'ac', replace it with 'cb'.
5. If the string prefix is 'ab', replace it with 'ca'.
6. If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.
7. If the string suffix is 'bb', delete the last two characters.
8. If the string starts with 'ac', replace the first two characters with 'zz'.

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Your output should be the final transformed string after applying all the rules.

Transform the following string according to the above list of rules:
acbaaaca

Answer: zzbaacbab
Metadata: {'string': 'acbaaaca', 'solution': 'zzbaacbab', 'states': ['acbaaaca', 'acbaaac', 'acbaacb', 'acbaacbab', 'zzbaacbab'], 'selected_rules': ["If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.", "If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.", "If the string ends with 'ca', remove the last character.", "If the string suffix is 'ac', replace it with 'cb'.", "If the string prefix is 'ab', replace it with 'ca'.", "If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.", "If the string suffix is 'bb', delete the last two characters.", "If the string starts with 'ac', replace the first two characters with 'zz'."], 'difficulty': {'string_length': 8, 'num_rules': 8}}

Example 2:
Question: Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
1. If the string suffix is 'bb', delete the last two characters.
2. If the string starts with 'bb', remove the second character.
3. If the string ends with 'aa', replace it with 'cc'.
4. If the string prefix is 'ab', replace it with 'ca'.
5. If the string ends with 'ca', remove the last character.
6. If the string contains 'bca', delete the first occurrence entirely.
7. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
8. If the string length is greater than 15, remove the middle character.

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Your output should be the final transformed string after applying all the rules.

Transform the following string according to the above list of rules:
bcabbc

Answer: bc
Metadata: {'string': 'bcabbc', 'solution': 'bc', 'states': ['bcabbc', 'bbc', 'bc'], 'selected_rules': ["If the string suffix is 'bb', delete the last two characters.", "If the string starts with 'bb', remove the second character.", "If the string ends with 'aa', replace it with 'cc'.", "If the string prefix is 'ab', replace it with 'ca'.", "If the string ends with 'ca', remove the last character.", "If the string contains 'bca', delete the first occurrence entirely.", "If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.", 'If the string length is greater than 15, remove the middle character.'], 'difficulty': {'string_length': 6, 'num_rules': 8}}

Example 3:
Question: Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
1. If the string contains 'acb', replace the first occurrence with its reverse ('bca').
2. If the string length is greater than 15, remove the middle character.
3. If the string starts with 'ac', replace the first two characters with 'zz'.
4. If the string ends with 'ba', replace it with 'ab'.
5. If the string starts with 'cc', remove the first two characters.
6. If the string suffix is 'ac', replace it with 'cb'.
7. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
8. If the string prefix is 'cb', replace it with 'aa' and delete the last character.

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Your output should be the final transformed string after applying all the rules.

Transform the following string according to the above list of rules:
cccaababaaacaaaccb

Answer: bbababcaaaccbc
Metadata: {'string': 'cccaababaaacaaaccb', 'solution': 'bbababcaaaccbc', 'states': ['cccaababaaacaaaccb', 'cccaababaacaaaccb', 'cccaababacaaaccb', 'cccaababcaaaccb', 'caababcaaaccb', 'bbababcaaaccbc'], 'selected_rules': ["If the string contains 'acb', replace the first occurrence with its reverse ('bca').", 'If the string length is greater than 15, remove the middle character.', "If the string starts with 'ac', replace the first two characters with 'zz'.", "If the string ends with 'ba', replace it with 'ab'.", "If the string starts with 'cc', remove the first two characters.", "If the string suffix is 'ac', replace it with 'cb'.", "If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.", "If the string prefix is 'cb', replace it with 'aa' and delete the last character."], 'difficulty': {'string_length': 18, 'num_rules': 8}}

````

### string_splitting
Generates String Splitting exercises with configurable difficulty

Default configuration:
```python
min_initial_machines = 0
max_initial_machines = 5
max_iterations = 1000
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Now, you have 5 machine A, 0 machine B, and 0 machine C. Provide the count of each machine and part type after applying the above rules.
Note: Apply the rules at most 1000 times. If the rules cannot be applied anymore, or if you have reached the maximum number of iterations, stop and provide the current counts of each machine and part type.

Answer: 0 0 0 5 0 5
Metadata: {'states': [[5, 0, 0, 0, 0, 0], [4, 0, 0, 2, 1, 0], [3, 0, 0, 4, 2, 0], [2, 0, 0, 6, 3, 0], [1, 0, 0, 8, 4, 0], [0, 0, 0, 10, 5, 0], [0, 0, 0, 9, 4, 1], [0, 0, 0, 8, 3, 2], [0, 0, 0, 7, 2, 3], [0, 0, 0, 6, 1, 4], [0, 0, 0, 5, 0, 5]], 'solution': [0, 0, 0, 5, 0, 5], 'difficulty': {'initial_machines': (5, 0, 0)}}

Example 2:
Question: There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Now, you have 0 machine A, 2 machine B, and 5 machine C. Provide the count of each machine and part type after applying the above rules.
Note: Apply the rules at most 1000 times. If the rules cannot be applied anymore, or if you have reached the maximum number of iterations, stop and provide the current counts of each machine and part type.

Answer: 0 0 1 0 1 1
Metadata: {'states': [[0, 2, 5, 0, 0, 0], [0, 0, 5, 1, 0, 0], [0, 0, 3, 1, 1, 0], [0, 0, 1, 1, 2, 0], [0, 0, 1, 0, 1, 1]], 'solution': [0, 0, 1, 0, 1, 1], 'difficulty': {'initial_machines': (0, 2, 5)}}

Example 3:
Question: There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Now, you have 3 machine A, 4 machine B, and 4 machine C. Provide the count of each machine and part type after applying the above rules.
Note: Apply the rules at most 1000 times. If the rules cannot be applied anymore, or if you have reached the maximum number of iterations, stop and provide the current counts of each machine and part type.

Answer: 0 0 0 3 0 5
Metadata: {'states': [[3, 4, 4, 0, 0, 0], [2, 4, 4, 2, 1, 0], [1, 4, 4, 4, 2, 0], [0, 4, 4, 6, 3, 0], [0, 2, 4, 7, 3, 0], [0, 0, 4, 8, 3, 0], [0, 0, 2, 8, 4, 0], [0, 0, 0, 8, 5, 0], [0, 0, 0, 7, 4, 1], [0, 0, 0, 6, 3, 2], [0, 0, 0, 5, 2, 3], [0, 0, 0, 4, 1, 4], [0, 0, 0, 3, 0, 5]], 'solution': [0, 0, 0, 3, 0, 5], 'difficulty': {'initial_machines': (3, 4, 4)}}

````

### string_synthesis
Generates String Synthesis exercises with configurable difficulty

Default configuration:
```python
min_initial_blocks = 0
max_initial_blocks = 5
max_iterations = 1000
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: There are nine different blocks [A] [B] [C] {A} {B} {C} (A) (B) (C)
1. One [A], one [B], and one [C] can be combined to form one {A}.
2. One [A] and one [B] can be combined to form one {C}.
3. One [B] and one [C] can be combined to form one {B}.
4. Two [C] can be combined to form one {C}.
5. One {A} and one {C} can be combined to form one (A) and one (B).
6. Two {B} can be combined to form one (C).

Given a certain number of initial blocks, your job is to cycle through the rules 1-6 above, synthesizing new blocks until no more rules can be applied, or until a state (counts of each block type) is repeated.
In the case a state is repeated the answer is the state before the repetition!

The output should be the count of each block type after the rules have been applied in the order they are listed above.
For example 1 0 3 0 2 0 0 0 1 means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C).

Now, you have 5 [A], 0 [B], and 0 [C] blocks. Provide the count of each block type after applying the above rules.
Note: Apply the rules at most 1000 times. If the rules cannot be applied anymore, or if you have reached the maximum number of iterations, stop and provide the current counts.

Answer: 5 0 0 0 0 0 0 0 0
Metadata: {'states': [[5, 0, 0, 0, 0, 0, 0, 0, 0]], 'solution': [5, 0, 0, 0, 0, 0, 0, 0, 0], 'difficulty': {'initial_blocks': (5, 0, 0)}}

Example 2:
Question: There are nine different blocks [A] [B] [C] {A} {B} {C} (A) (B) (C)
1. One [A], one [B], and one [C] can be combined to form one {A}.
2. One [A] and one [B] can be combined to form one {C}.
3. One [B] and one [C] can be combined to form one {B}.
4. Two [C] can be combined to form one {C}.
5. One {A} and one {C} can be combined to form one (A) and one (B).
6. Two {B} can be combined to form one (C).

Given a certain number of initial blocks, your job is to cycle through the rules 1-6 above, synthesizing new blocks until no more rules can be applied, or until a state (counts of each block type) is repeated.
In the case a state is repeated the answer is the state before the repetition!

The output should be the count of each block type after the rules have been applied in the order they are listed above.
For example 1 0 3 0 2 0 0 0 1 means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C).

Now, you have 0 [A], 2 [B], and 5 [C] blocks. Provide the count of each block type after applying the above rules.
Note: Apply the rules at most 1000 times. If the rules cannot be applied anymore, or if you have reached the maximum number of iterations, stop and provide the current counts.

Answer: 0 0 1 0 0 1 0 0 1
Metadata: {'states': [[0, 2, 5, 0, 0, 0, 0, 0, 0], [0, 1, 4, 0, 1, 0, 0, 0, 0], [0, 0, 3, 0, 2, 0, 0, 0, 0], [0, 0, 1, 0, 2, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1]], 'solution': [0, 0, 1, 0, 0, 1, 0, 0, 1], 'difficulty': {'initial_blocks': (0, 2, 5)}}

Example 3:
Question: There are nine different blocks [A] [B] [C] {A} {B} {C} (A) (B) (C)
1. One [A], one [B], and one [C] can be combined to form one {A}.
2. One [A] and one [B] can be combined to form one {C}.
3. One [B] and one [C] can be combined to form one {B}.
4. Two [C] can be combined to form one {C}.
5. One {A} and one {C} can be combined to form one (A) and one (B).
6. Two {B} can be combined to form one (C).

Given a certain number of initial blocks, your job is to cycle through the rules 1-6 above, synthesizing new blocks until no more rules can be applied, or until a state (counts of each block type) is repeated.
In the case a state is repeated the answer is the state before the repetition!

The output should be the count of each block type after the rules have been applied in the order they are listed above.
For example 1 0 3 0 2 0 0 0 1 means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C).

Now, you have 3 [A], 4 [B], and 4 [C] blocks. Provide the count of each block type after applying the above rules.
Note: Apply the rules at most 1000 times. If the rules cannot be applied anymore, or if you have reached the maximum number of iterations, stop and provide the current counts.

Answer: 0 0 0 3 1 0 0 0 0
Metadata: {'states': [[3, 4, 4, 0, 0, 0, 0, 0, 0], [2, 3, 3, 1, 0, 0, 0, 0, 0], [1, 2, 2, 2, 0, 0, 0, 0, 0], [0, 1, 1, 3, 0, 0, 0, 0, 0], [0, 0, 0, 3, 1, 0, 0, 0, 0]], 'solution': [0, 0, 0, 3, 1, 0, 0, 0, 0], 'difficulty': {'initial_blocks': (3, 4, 4)}}

````

### sudoku
Generates sudoku puzzles with configurable difficulty

Default configuration:
```python
min_empty = 30
max_empty = 50
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve this Sudoku puzzle:
4 _ _ _ 9 2 _ 3 _
_ _ 3 4 6 _ _ _ 7
6 1 2 _ _ 7 8 _ _
2 _ _ _ _ _ 7 9 1
8 _ _ 7 1 _ _ 5 6
1 _ _ 5 _ _ _ _ 3
9 _ 4 _ 7 1 _ _ _
_ 8 _ _ _ _ _ _ _
_ _ _ 9 8 _ _ _ 4
Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines.
Answer: 4 7 8 1 9 2 6 3 5
5 9 3 4 6 8 1 2 7
6 1 2 3 5 7 8 4 9
2 4 5 8 3 6 7 9 1
8 3 9 7 1 4 2 5 6
1 6 7 5 2 9 4 8 3
9 5 4 2 7 1 3 6 8
3 8 1 6 4 5 9 7 2
7 2 6 9 8 3 5 1 4
Metadata: {'puzzle': [[4, 0, 0, 0, 9, 2, 0, 3, 0], [0, 0, 3, 4, 6, 0, 0, 0, 7], [6, 1, 2, 0, 0, 7, 8, 0, 0], [2, 0, 0, 0, 0, 0, 7, 9, 1], [8, 0, 0, 7, 1, 0, 0, 5, 6], [1, 0, 0, 5, 0, 0, 0, 0, 3], [9, 0, 4, 0, 7, 1, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 9, 8, 0, 0, 0, 4]], 'solution': [[4, 7, 8, 1, 9, 2, 6, 3, 5], [5, 9, 3, 4, 6, 8, 1, 2, 7], [6, 1, 2, 3, 5, 7, 8, 4, 9], [2, 4, 5, 8, 3, 6, 7, 9, 1], [8, 3, 9, 7, 1, 4, 2, 5, 6], [1, 6, 7, 5, 2, 9, 4, 8, 3], [9, 5, 4, 2, 7, 1, 3, 6, 8], [3, 8, 1, 6, 4, 5, 9, 7, 2], [7, 2, 6, 9, 8, 3, 5, 1, 4]], 'num_empty': 48, 'difficulty': {'num_empty': 48}}

Example 2:
Question: Solve this Sudoku puzzle:
_ _ _ 1 3 2 6 4 5
_ 4 _ 8 5 _ _ 9 _
_ _ 1 9 _ 7 _ _ _
_ 8 9 6 _ _ 7 5 4
_ 3 _ 4 _ 1 9 8 _
4 6 _ 5 9 _ 2 3 1
5 _ 4 7 1 9 3 _ _
9 7 6 _ _ 4 5 1 _
8 _ 3 _ _ _ 4 7 _
Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines.
Answer: 7 9 8 1 3 2 6 4 5
3 4 2 8 5 6 1 9 7
6 5 1 9 4 7 8 2 3
1 8 9 6 2 3 7 5 4
2 3 5 4 7 1 9 8 6
4 6 7 5 9 8 2 3 1
5 2 4 7 1 9 3 6 8
9 7 6 3 8 4 5 1 2
8 1 3 2 6 5 4 7 9
Metadata: {'puzzle': [[0, 0, 0, 1, 3, 2, 6, 4, 5], [0, 4, 0, 8, 5, 0, 0, 9, 0], [0, 0, 1, 9, 0, 7, 0, 0, 0], [0, 8, 9, 6, 0, 0, 7, 5, 4], [0, 3, 0, 4, 0, 1, 9, 8, 0], [4, 6, 0, 5, 9, 0, 2, 3, 1], [5, 0, 4, 7, 1, 9, 3, 0, 0], [9, 7, 6, 0, 0, 4, 5, 1, 0], [8, 0, 3, 0, 0, 0, 4, 7, 0]], 'solution': [[7, 9, 8, 1, 3, 2, 6, 4, 5], [3, 4, 2, 8, 5, 6, 1, 9, 7], [6, 5, 1, 9, 4, 7, 8, 2, 3], [1, 8, 9, 6, 2, 3, 7, 5, 4], [2, 3, 5, 4, 7, 1, 9, 8, 6], [4, 6, 7, 5, 9, 8, 2, 3, 1], [5, 2, 4, 7, 1, 9, 3, 6, 8], [9, 7, 6, 3, 8, 4, 5, 1, 2], [8, 1, 3, 2, 6, 5, 4, 7, 9]], 'num_empty': 34, 'difficulty': {'num_empty': 34}}

Example 3:
Question: Solve this Sudoku puzzle:
_ _ 1 9 2 _ _ _ 3
3 _ _ 1 7 5 8 2 6
_ _ _ 4 3 6 1 _ _
1 _ 5 7 _ _ 9 3 _
_ 4 _ _ 5 9 7 1 8
7 _ 9 _ 1 _ 6 4 5
_ _ 3 5 9 _ 2 8 4
_ _ 2 6 8 _ _ 9 1
_ 5 _ 2 4 1 3 _ _
Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines.
Answer: 5 6 1 9 2 8 4 7 3
3 9 4 1 7 5 8 2 6
8 2 7 4 3 6 1 5 9
1 8 5 7 6 4 9 3 2
2 4 6 3 5 9 7 1 8
7 3 9 8 1 2 6 4 5
6 1 3 5 9 7 2 8 4
4 7 2 6 8 3 5 9 1
9 5 8 2 4 1 3 6 7
Metadata: {'puzzle': [[0, 0, 1, 9, 2, 0, 0, 0, 3], [3, 0, 0, 1, 7, 5, 8, 2, 6], [0, 0, 0, 4, 3, 6, 1, 0, 0], [1, 0, 5, 7, 0, 0, 9, 3, 0], [0, 4, 0, 0, 5, 9, 7, 1, 8], [7, 0, 9, 0, 1, 0, 6, 4, 5], [0, 0, 3, 5, 9, 0, 2, 8, 4], [0, 0, 2, 6, 8, 0, 0, 9, 1], [0, 5, 0, 2, 4, 1, 3, 0, 0]], 'solution': [[5, 6, 1, 9, 2, 8, 4, 7, 3], [3, 9, 4, 1, 7, 5, 8, 2, 6], [8, 2, 7, 4, 3, 6, 1, 5, 9], [1, 8, 5, 7, 6, 4, 9, 3, 2], [2, 4, 6, 3, 5, 9, 7, 1, 8], [7, 3, 9, 8, 1, 2, 6, 4, 5], [6, 1, 3, 5, 9, 7, 2, 8, 4], [4, 7, 2, 6, 8, 3, 5, 9, 1], [9, 5, 8, 2, 4, 1, 3, 6, 7]], 'num_empty': 33, 'difficulty': {'num_empty': 33}}

````

### syllogism
Generates syllogism reasoning tasks

Default configuration:
```python
allow_all = True
allow_no = True
allow_some = True
allow_some_not = True
invalid_ratio = 0.3
inversion_probability = 0.3
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Consider these statements:
1. No students are humans
2. All humans are chefs

Does it logically follow that:
Some chefs are humans?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'No students are humans', 'premise2': 'All humans are chefs', 'selected_premise': 2, 'conclusion': 'Some chefs are humans', 'is_valid': True, 'type': 'inversion'}

Example 2:
Question: Consider these statements:
1. All children are animals
2. Some animals are not doctors

Does it logically follow that:
Some children are not doctors?
(Answer Yes or No)
Answer: Yes
Metadata: {'premise1': 'All children are animals', 'premise2': 'Some animals are not doctors', 'conclusion': 'Some children are not doctors', 'is_valid': True, 'type': 'syllogism'}

Example 3:
Question: Consider these statements:
1. Some butterflies are not tigers
2. No tigers are whales

Does it logically follow that:
Some butterflies are whales?
(Answer Yes or No)
Answer: No
Metadata: {'premise1': 'Some butterflies are not tigers', 'premise2': 'No tigers are whales', 'conclusion': 'Some butterflies are whales', 'is_valid': False, 'type': 'syllogism'}

````

### time_intervals
Generates time interval calculation tasks with various formats and complexities

Default configuration:
```python
min_time = 00:00:00
max_time = 23:59:59.999999
max_time_difference_seconds = 86400
min_date = 1900-01-01
max_date = 3000-01-01
max_date_difference_days = 100
task_types = ['time', 'time_seconds', 'time_ms', 'date', 'datetime', 'datetime_tz']
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: A system backup started at 2964-06-17 08:15:14 and completed at 2964-07-04 11:59:09. What was the total backup duration? Answer in D days, HH:MM.
Answer: 17 days, 03:43
Metadata: {'task_type': 'datetime_tz', 'start_time': datetime.datetime(2964, 6, 17, 8, 15, 14), 'end_time': datetime.datetime(2964, 7, 4, 11, 59, 9), 'format': '%Y-%m-%d %H:%M:%S', 'expected_format': 'D days, HH:MM', 'difficulty': {'max_time_difference_seconds': 86400, 'max_date_difference_days': 100}}

Example 2:
Question: A video call started at 09:44 and ended at 12:22. How long was the call? Answer in HH:MM.
Answer: 02:38
Metadata: {'task_type': 'time', 'start_time': datetime.datetime(2025, 3, 16, 9, 44), 'end_time': datetime.datetime(2025, 3, 16, 12, 22), 'format': '%H:%M', 'expected_format': 'HH:MM', 'difficulty': {'max_time_difference_seconds': 86400, 'max_date_difference_days': 100}}

Example 3:
Question: Calculate the time difference between Sat Dec 22 2677 and Thu Mar 21 2678. Express the result in D days.
Answer: 89 days
Metadata: {'task_type': 'date', 'start_time': datetime.datetime(2677, 12, 22, 0, 0), 'end_time': datetime.datetime(2678, 3, 21, 0, 0), 'format': '%a %b %d %Y', 'expected_format': 'D days', 'difficulty': {'max_time_difference_seconds': 86400, 'max_date_difference_days': 100}}

````

### tower_of_hanoi
Generates Tower of Hanoi problems with solutions.
    Supports variable number of pegs using the optimized Frame-Stewart algorithm with Peg State Tracking.

Default configuration:
```python
min_disks = 3
max_disks = 7
min_pegs = 3
max_pegs = 4
size = 500
seed = 42
visualize = False
```

Example tasks:
````
Example 1:
Question: Solve the Tower of Hanoi problem with 3 disks and 3 pegs.
Move all disks from Peg 3 to Peg 2 following the rules:
- Only one disk can be moved at a time.
- A larger disk cannot be placed on top of a smaller disk.
- All disks must be on a peg at all times.

Provide the sequence of moves.

Formatting guidelines:
- Each instruction should be placed on a single line.
- Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
- Do not include any other text or formatting.

Answer: Move disk 1 from Peg 3 to Peg 2
Move disk 2 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 3 from Peg 3 to Peg 2
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Metadata: {'num_disks': 3, 'num_pegs': 3, 'start_peg': 3, 'target_peg': 2, 'auxiliary_pegs': [1], 'solution_length': 7}

Example 2:
Question: Solve the Tower of Hanoi problem with 3 disks and 4 pegs.
Move all disks from Peg 2 to Peg 4 following the rules:
- Only one disk can be moved at a time.
- A larger disk cannot be placed on top of a smaller disk.
- All disks must be on a peg at all times.

Provide the sequence of moves.

Formatting guidelines:
- Each instruction should be placed on a single line.
- Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
- Do not include any other text or formatting.

Answer: Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 3 from Peg 2 to Peg 4
Move disk 2 from Peg 3 to Peg 4
Move disk 1 from Peg 1 to Peg 4
Metadata: {'num_disks': 3, 'num_pegs': 4, 'start_peg': 2, 'target_peg': 4, 'auxiliary_pegs': [1, 3], 'solution_length': 5}

Example 3:
Question: Solve the Tower of Hanoi problem with 6 disks and 3 pegs.
Move all disks from Peg 1 to Peg 2 following the rules:
- Only one disk can be moved at a time.
- A larger disk cannot be placed on top of a smaller disk.
- All disks must be on a peg at all times.

Provide the sequence of moves.

Formatting guidelines:
- Each instruction should be placed on a single line.
- Each line should be formatted as 'Move disk X from Peg Y to Peg Z'
- Do not include any other text or formatting.

Answer: Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 3 from Peg 1 to Peg 3
Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 4 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 2 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 3 from Peg 3 to Peg 2
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 5 from Peg 1 to Peg 3
Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 3 from Peg 2 to Peg 1
Move disk 1 from Peg 3 to Peg 2
Move disk 2 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 4 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 3 from Peg 1 to Peg 3
Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 6 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 2 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 3 from Peg 3 to Peg 2
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 4 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 3 from Peg 2 to Peg 1
Move disk 1 from Peg 3 to Peg 2
Move disk 2 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 5 from Peg 3 to Peg 2
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 3 from Peg 1 to Peg 3
Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 4 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 2 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 3 from Peg 3 to Peg 2
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Metadata: {'num_disks': 6, 'num_pegs': 3, 'start_peg': 1, 'target_peg': 2, 'auxiliary_pegs': [3], 'solution_length': 63}

````

### tsumego
Generates Tsumego problems with configurable parameters

Default configuration:
```python
min_board_size = 9
max_board_size = 13
max_stones = 15
size = 500
seed = 42
```

Example tasks:
````
Example 1:
Question: I have a Go problem for you. Black moves next - can you capture some of the white stones?

```
   A B C D E F G H I
 9 X . . . X . . . .
 8 . . . . . . . . .
 7 . O . O . . X . .
 6 . . . X . . . . O
 5 O . X O X . . . .
 4 . X O O . O . . .
 3 . . X O X . . . .
 2 . . . X . . . . .
 1 . O . O . . X . .
```

X - Black
O - White

Specify your move in coordinates (e.g. 'C4' for column C, row 4)
Answer: E4
Metadata: {'difficulty': {'board_size': 9}, 'board': [['X', '.', '.', '.', 'X', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', 'O', '.', 'O', '.', '.', 'X', '.', '.'], ['.', '.', '.', 'X', '.', '.', '.', '.', 'O'], ['O', '.', 'X', 'O', 'X', '.', '.', '.', '.'], ['.', 'X', 'O', 'O', '.', 'O', '.', '.', '.'], ['.', '.', 'X', 'O', 'X', '.', '.', '.', '.'], ['.', '.', '.', 'X', '.', '.', '.', '.', '.'], ['.', 'O', '.', 'O', '.', '.', 'X', '.', '.']]}

Example 2:
Question: Here's a Go challenge. Playing as Black, how can you capture as many white stones as possible?

```
   A B C D E F G H I
 9 . . O . . . . . .
 8 . X O . . . . . .
 7 X . X . . . . . .
 6 O O O X . . . . .
 5 X O O . . . . . .
 4 . X . . . . . . O
 3 . X . . . . X . .
 2 O . O . . . . . .
 1 . . . . O . . . .
```

X - Black
O - White

Specify your move in coordinates (e.g. 'C4' for column C, row 4)
Answer: B7
Metadata: {'difficulty': {'board_size': 9}, 'board': [['.', '.', 'O', '.', '.', '.', '.', '.', '.'], ['.', 'X', 'O', '.', '.', '.', '.', '.', '.'], ['X', '.', 'X', '.', '.', '.', '.', '.', '.'], ['O', 'O', 'O', 'X', '.', '.', '.', '.', '.'], ['X', 'O', 'O', '.', '.', '.', '.', '.', '.'], ['.', 'X', '.', '.', '.', '.', '.', '.', 'O'], ['.', 'X', '.', '.', '.', '.', 'X', '.', '.'], ['O', '.', 'O', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', 'O', '.', '.', '.', '.']]}

Example 3:
Question: Tsumego time. Black to play and capture some stones.
Find the key move.

```
   A B C D E F G H I J K L
12 . . . . . . . . . . . .
11 . . X . . . . . . . . .
10 . . . . . . . . . . . .
 9 . . . . . . . . . . . .
 8 X . . . . X . . . X . .
 7 . X . . . . . . . . . .
 6 . O X X . . . . . . . O
 5 . X O O X . . . . . . .
 4 . O O . . . . . O . . O
 3 X . X . . . . . . . . .
 2 . . . . . . . . . . . .
 1 . . . . . . . . . . X .
```

X - Black
O - White

Specify your move in coordinates (e.g. 'C4' for column C, row 4)
Answer: D4
Metadata: {'difficulty': {'board_size': 12}, 'board': [['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', 'X', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['X', '.', '.', '.', '.', 'X', '.', '.', '.', 'X', '.', '.'], ['.', 'X', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', 'O', 'X', 'X', '.', '.', '.', '.', '.', '.', '.', 'O'], ['.', 'X', 'O', 'O', 'X', '.', '.', '.', '.', '.', '.', '.'], ['.', 'O', 'O', '.', '.', '.', '.', '.', 'O', '.', '.', 'O'], ['X', '.', 'X', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'X', '.']]}

````

### word_ladder
Generates word ladder transformation tasks

Default configuration:
```python
min_word_length = 4
max_word_length = 4
min_chain_length = -1
max_chain_length = -1
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Transform the word ladder 'HAND' to 'GLEE' by changing one letter at a time.
Provide your answer as a comma-separated sequence of uppercase letters without spaces.
Each step must be a valid English word.
Answer: HAND,LAND,LEND,FEND,FEED,FLED,FLEE,GLEE
Metadata: {'start_word': 'HAND', 'end_word': 'GLEE', 'word_length': 4, 'chain_length': 8, 'difficulty': {'word_length': 4, 'chain_length': 8}}

Example 2:
Question: Transform the word ladder 'JAZZ' to 'DORM' by changing one letter at a time.
Provide your answer as a comma-separated sequence of uppercase letters without spaces.
Each step must be a valid English word.
Answer: JAZZ,JIZZ,FIZZ,FUZZ,FUZE,FAZE,FARE,FORE,FORM,DORM
Metadata: {'start_word': 'JAZZ', 'end_word': 'DORM', 'word_length': 4, 'chain_length': 10, 'difficulty': {'word_length': 4, 'chain_length': 10}}

Example 3:
Question: Transform the word ladder 'SNOG' to 'SUQS' by changing one letter at a time.
Provide your answer as a comma-separated sequence of uppercase letters without spaces.
Each step must be a valid English word.
Answer: SNOG,SNOW,SHOW,SHEW,SHES,SUES,SUQS
Metadata: {'start_word': 'SNOG', 'end_word': 'SUQS', 'word_length': 4, 'chain_length': 7, 'difficulty': {'word_length': 4, 'chain_length': 7}}

````

### word_sequence_reversal
Generates word sequence reversal tasks from text spans

Default configuration:
```python
min_words = 3
max_words = 8
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Solve the following problem.

Provide you answer as a comma-separated list of words with a space after the comma.

Reverse this list of words: bed, if, problem, but, Well, an, transmission, nutritive

Answer: nutritive, transmission, an, Well, but, problem, if, bed
Metadata: {'num_words': 8, 'words': ['bed', 'if', 'problem', 'but', 'Well', 'an', 'transmission', 'nutritive'], 'difficulty': {'words': 8}}

Example 2:
Question: Solve the following problem.

Provide you answer as a comma-separated list of words with a space after the comma.

Reverse this list of words: it, pleasure, Gutenberg

Answer: Gutenberg, pleasure, it
Metadata: {'num_words': 3, 'words': ['it', 'pleasure', 'Gutenberg'], 'difficulty': {'words': 3}}

Example 3:
Question: Solve the following problem.

Provide you answer as a comma-separated list of words with a space after the comma.

Reverse this list of words: readable, to, he, that, to, possession

Answer: possession, to, that, he, to, readable
Metadata: {'num_words': 6, 'words': ['readable', 'to', 'he', 'that', 'to', 'possession'], 'difficulty': {'words': 6}}

````

### word_sorting
Generates word sorting tasks

Default configuration:
```python
min_words = 3
max_words = 10
min_word_length = 3
max_word_length = 12
transformation = original
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.

Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3

Now, sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: DIRECT, given, exclaims, dreaming

Answer: DIRECT, dreaming, exclaims, given
Metadata: {'difficulty': {'num_words': 4, 'word_length': 8}, 'original_words': ['DIRECT', 'given', 'exclaims', 'dreaming'], 'sorted_words': ['DIRECT', 'dreaming', 'exclaims', 'given'], 'transformed_words': ['DIRECT', 'given', 'exclaims', 'dreaming'], 'direction': 'ascending'}

Example 2:
Question: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.

Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3

Now, sort these words in descending order (using ASCII/Unicode ordering) and return them as a comma-separated list: heat, begun, sometimes

Answer: sometimes, heat, begun
Metadata: {'difficulty': {'num_words': 3, 'word_length': 9}, 'original_words': ['heat', 'begun', 'sometimes'], 'sorted_words': ['sometimes', 'heat', 'begun'], 'transformed_words': ['heat', 'begun', 'sometimes'], 'direction': 'descending'}

Example 3:
Question: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.

Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3

Now, sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: violates, yes, already, completing, pages, duty, his, EXPRESS, duly

Answer: EXPRESS, already, completing, duly, duty, his, pages, violates, yes
Metadata: {'difficulty': {'num_words': 9, 'word_length': 10}, 'original_words': ['violates', 'yes', 'already', 'completing', 'pages', 'duty', 'his', 'EXPRESS', 'duly'], 'sorted_words': ['EXPRESS', 'already', 'completing', 'duly', 'duty', 'his', 'pages', 'violates', 'yes'], 'transformed_words': ['violates', 'yes', 'already', 'completing', 'pages', 'duty', 'his', 'EXPRESS', 'duly'], 'direction': 'ascending'}

````

### zebra_puzzles
Generates [Zebra Puzzles](https://en.wikipedia.org/wiki/Zebra_Puzzle) with configurable parameters

Default configuration:
```python
num_people = 4
num_characteristics = 4
seed = 42
size = 500
```

Example tasks:
````
Example 1:
Question: This is a logic puzzle. There are 4 houses (numbered 1 on the left, 4 on the right), from the perspective of someone standing across the street from them. Each has a different person in them. They have different characteristics:
 - Each person has a unique name: carol, arnold, alice, bob
 - People use different phone models: huawei p50, samsung galaxy s21, oneplus 9, google pixel 6
 - Each person has a favorite drink: milk, boba tea, coffee, water
 - The people keep different animals: bird, cat, fish, dog

1. Alice is the cat lover.
2. The person who likes milk is in the third house.
3. The person who uses a Huawei P50 is Bob.
4. The one who only drinks water is the bird keeper.
5. The cat lover is in the second house.
6. The boba tea drinker is the dog owner.
7. The person who uses a Google Pixel 6 is directly left of Carol.
8. The one who only drinks water is Carol.
9. Carol is the person who uses a OnePlus 9.

What is Name of the person who lives in House 1?? Provide only the name of the person as your final answer.
Answer: bob
Metadata: {'difficulty': {'num_people': 4, 'num_characteristics': 4}}

Example 2:
Question: This is a logic puzzle. There are 4 houses (numbered 1 on the left, 4 on the right), from the perspective of someone standing across the street from them. Each has a different person in them. They have different characteristics:
 - Each person has a unique name: alice, bob, arnold, carol
 - Each mother is accompanied by their child: alice, bella, billy, timothy
 - The people are of nationalities: brit, german, chinese, dane
 - Everyone has something different for lunch: soup, stir fry, grilled cheese, pizza

1. The British person is Arnold.
2. The person's child is named Alice is directly left of the person who loves the soup.
3. The person who loves stir fry is the person's child is named Bella.
4. The Chinese is Carol.
5. The German is the person's child is named Bella.
6. The person's child is named Bella is Bob.
7. The person who loves the soup is in the second house.
8. The person who loves the soup is the British person.
9. The person's child is named Alice is Carol.
10. The British person is directly left of the German.
11. The person who is the mother of Billy is the person who is a pizza lover.

What is Name of the person who lives in House 1?? Provide only the name of the person as your final answer.
Answer: carol
Metadata: {'difficulty': {'num_people': 4, 'num_characteristics': 4}}

Example 3:
Question: This is a logic puzzle. There are 4 houses (numbered 1 on the left, 4 on the right), from the perspective of someone standing across the street from them. Each has a different person in them. They have different characteristics:
 - Each person has a unique name: alice, arnold, bob, carol
 - Everyone has a different favorite cigar: pall mall, dunhill, blue master, prince
 - Everyone has something different for lunch: stir fry, grilled cheese, soup, pizza
 - Each person has a favorite color: blue, purple, brown, white

1. The person who loves white is the person who loves stir fry.
2. The person who loves brown is directly left of the Prince smoker.
3. The person who is a pizza lover and Arnold are next to each other.
4. The person partial to Pall Mall is the person who loves white.
5. Alice is the person who loves the soup.
6. The person partial to Pall Mall is directly left of the person who loves the soup.
7. The person who smokes Blue Master is directly left of the Dunhill smoker.
8. The Dunhill smoker is Bob.
9. The person who loves the soup is the person who loves blue.

What is Name of the person who lives in House 1?? Provide only the name of the person as your final answer.
Answer: carol
Metadata: {'difficulty': {'num_people': 4, 'num_characteristics': 4}}

````


