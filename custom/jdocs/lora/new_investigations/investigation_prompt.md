now we are facing unexpected bad groot finetuned model inference performance issue. that is the symptom, but root cause could be somewhere in the whole pipeline: data collection(calibration, verification), training data conversion, training process, inference process. the hardness to solve the issue is there is a long chain for the process, any issue in the pipeline could cause the catestraphic result at inference stage, so we have to review the whole process module by module, verify in terms of setup experiments(ie: use data inputs, and expected outputs to verify against the real output). 

our current used scripts:
- data collection script: /home/jrobot/project/XLeRobot/scripts/collect_xlerobot_data.py
- v3 to v2 groot compatible data conversion script: /home/jrobot/project/Isaac-GR00T/custom/scripts/convert_lerobot_v3_to_groot.py
- training script: /home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_mvp.sh
- inference script: /home/jrobot/project/Isaac-GR00T/custom/scripts/infer_groot_async.py
- collected v3 dataset: /home/jrobot/project/XLeRobot/datasets/left/pick_and_place
- converted v2 dataset: /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place
- latest trained model: /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_230404914

the fundamental mindset is we have to use first principle mindset to inspect each key component along the chain in terms of setup experiments:
- draw a whole picture for the pipeline in mermaid: including hardware, dataset, model, etc
- for each key modules: data collection, data conversion, data loading, training, inference
    - draw its mermaid diagram with an example inputs outputs data which works as the high level understanding and point out the potential issue we could face 
    - experiment setup to verify
        - expected inputs and outputs with supported facts or logic
        - setup experiments 
        - run experiments to verify the inputs output against the "successfully finetuned models and their datasets" mentioned below(you can use "example 1" as the major comparison counterpart, but you can check other examples to cross check )

useful reference and resources:
- we have a working groot tutorial, which open sourced dataset, finetuning script, open loop eval and inference demo, we should compare ours with theirs one by one to spot the issue in our current pipeline:
    - verify data loading against: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/0_load_dataset.ipynb
        - load data
        - Show Image frames within the data
        - Transforming the data
        - verify and check how the data is different after applying the transformations
    - verify inference process against https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/1_gr00t_inference.ipynb
        - Loading Pretrained Policy
        - Loading dataset
            - verify embodiment tags used in pretrained policy
            - I found that we new "new embodiment" for inference, but I didn't find any mentioning of this in training script?
        - print single step_data and visualize it
        - run the policy from the pretrained checkpoint on the step data and verify it
    - finetuning process against: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/2_finetuning.ipynb
        - embodiment_tags: what is its purpose, what should we use? what we already used?
        - do we need to use sim data to verify the data collection, transformation and training process, as now our own dataset based finetuning has much worse performance than what open loop eval shown at the end.
        - we need to reproduce what in https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/getting_started/3_0_new_embodiment_finetuning.md
    - [IMPORTANT] deep diving into our current settings where we could easily missed something or incorrectly used something: https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/getting_started/4_deeper_understanding.md
        - embodiment tags meaning and how should we choose for so101 arm and our collected data for finetuning
        - for tuning parameter: is tune_projector enough for our finetuning?
        - Data Transforms: did we miss something? how to verify?
    - data conversion check against: https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/getting_started/LeRobot_compatible_data_schema.md
        - could need to compare with demo_data/robot_sim.PickNPlace data to inspect
    - groot tutorial speicified in https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/3_0_new_embodiment_finetuning.md locally, then inspect it and compare with ours to see if there are any issues in our datasets
    
- we have lerobot resources
    - training:
        - https://huggingface.co/docs/lerobot/il_robots#train-a-policy
        - https://huggingface.co/docs/lerobot/bring_your_own_policies
        - https://huggingface.co/docs/lerobot/integrate_hardware
        - https://huggingface.co/docs/lerobot/hilserl_sim
        - https://huggingface.co/docs/lerobot/lerobot-dataset-v3
        - https://huggingface.co/docs/lerobot/porting_datasets_v3
        - https://huggingface.co/docs/lerobot/using_dataset_tools
        - https://huggingface.co/docs/lerobot/groot
        - https://huggingface.co/docs/lerobot/async
        - https://huggingface.co/docs/lerobot/rtc
        - https://huggingface.co/docs/lerobot/envhub_leisaac
        - https://huggingface.co/docs/lerobot/libero
    - robot processors:
        - https://huggingface.co/docs/lerobot/introduction_processors
            - key insights:
                - a fundamental mismatch between the data that robots and humans produce and what machine learning models expect
                - model outputs mismatch
                - Cross-domain translation
        - https://huggingface.co/docs/lerobot/debug_processor_pipeline
        - https://huggingface.co/docs/lerobot/implement_your_own_processor
        - https://huggingface.co/docs/lerobot/processors_robots_teleop
        - https://huggingface.co/docs/lerobot/env_processor

- check against successfully finetuned models and their datasets:
    - example 1: https://huggingface.co/Pushpakcc/gr00t-so100_dualcam-finetuned
        - compare its datasets with ours to spot any data collection issue: https://huggingface.co/datasets/youliangtan/so101-table-cleanup
        - its Embodiment: SO-101 robot arm with 6-DOF control, Embodiment Head: Custom action head for SO-101 robot configuration
        - are we using dual-camera modality file?
    - example 2: https://huggingface.co/5hadytru/so101_GR00T-N1.5-3B_v3
        - compare its datasets with ours to spot any data collection issue: https://huggingface.co/datasets/5hadytru/so101_grasp_1
    - example 3: https://huggingface.co/c299m/so101-pen-in-box-v2-policy
        - compare its datasets with outs: https://huggingface.co/datasets/c299m/so101-pen-in-box-v2


useful tools: 
- visualization: 
    - visualize dataset locally: https://wiki.seeedstudio.com/lerobot_so100m_new/#visualize-the-dataset
    - replay an episode: https://wiki.seeedstudio.com/lerobot_so100m_new/#replay-an-episode


reference resources:
- https://xlerobot.readthedocs.io/en/latest/software/getting_started/RL_VLA.html#vision-language-action-vla-training-for-xlerobot
    - hardware verification
    - data collection comparison: their script and our script
    
you can use folder custom/jdocs/lora/new_investigations for this new research: use subagents to help do research and assign tasks, you can put researched information and insights into custom/jdocs/lora/new_investigations/, which can be used later to reference without have to do research again, they can also be used for you to synthesize and organize into the final investigation report: custom/jdocs/lora/new_investigations/gemini_final_research_report_and_action_plan.md.


