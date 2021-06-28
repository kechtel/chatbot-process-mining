# Chatbot Process Mining

![example workflow](https://github.com/kechtel/chatbot-process-mining/actions/workflows/python-app.yml/badge.svg)

This repository contains the source code for my master's thesis entitled "Evaluating Chatbots' Ability to Learn Business Processes".
For a more comprehensive overview of the approach and the steps invoked in the scripts listed below, I refer to the thesis.


## Installation

To checkout the repository, create a new Python virtual environment, and install the dependencies, execute the following commands:

```
git clone https://github.com/kechtel/chatbot-process-mining
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Evaluating

The file `chatbot_evaluation.py` evaluates a chatbot by processing the event logs of conversations used for training the chatbot and the replayed conversations with the chatbot.
If executed without parameters, it searches for directories with the name of the organization the chatbot was trained for in the path `xes/` and for the corresponding four event logs in the directory of the respective organization.
The discovered proxy process models are saved to the directory `models`, whereas the evaluation results are saved as Excel files in the current directory.
To run the file, execute the following command:

```
python3 chatbot_evaluation.py
```

The following optional parameters can additionally be specified:

* `--training-event-log`: filename of the `training_event_log`
* `--training-replayed-event-log`: filename of the \newline `training_replayed_event_log`
* `--test-event-log`: filename of the `test_event_log`
* `--test-replayed-event-log`: filename of the `test_replayed_event_log`
* `--variants-filter`: PM4Py variants filter (`variants_percentage_x.x` or `variants_auto_x.x` or `variants_top_k`)
* `--company`: Company whose chatbot is analyzed
* `--model-file`: `.pnml` file of normative process model
