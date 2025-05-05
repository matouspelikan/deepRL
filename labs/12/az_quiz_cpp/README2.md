Dear all,

here you can find the solutions of the az_quiz_cpp assignment:

az_quiz_agent_cpp.solution.py, az_quiz.h.solution.py, az_quiz_mcts.h.solution.py, az_quiz_sim_game.h.solution.py: a more advanced C++ agent for AZQuiz. It is based on the TensorFlow implementation az_quiz_agent.tf.solution.py, but
the mcts and sim_game are implemented in C++
additionally, rotations (and also other symmetries) are supported (when using the option --rotate); in that case, each board is rotated/mirrored randomly during training and batch prediction (but not during a prediction of a single example); enabling this option make training considerably faster
Cheers,
Milan S.