# cognitive_neuro_bot.py
# A simple domain-specific chatbot: Cognitive Neuroscience Assistant
# CLI usage: python cognitive_neuro_bot.py

import re
import sys
import json
import math
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1) Comprehensive domain knowledge base
# -------------------------------
KB_DOCS = [
    {
        "title": "Neurotransmitters: Dopamine",
        "text": """
Dopamine is a catecholamine neurotransmitter crucial for reward processing, motivation, motor control, and executive functions. Synthesized from tyrosine via L-DOPA, it acts on five receptor subtypes (D1-D5) classified into D1-like (D1, D5) and D2-like (D2, D3, D4) families. Major pathways: mesolimbic (VTA to nucleus accumbens; reward/motivation), mesocortical (VTA to prefrontal cortex; cognition), nigrostriatal (substantia nigra to striatum; motor control), and tuberoinfundibular (hypothalamus to pituitary; hormone regulation). Dysfunction implicated in Parkinson's disease (nigrostriatal degeneration), ADHD (prefrontal hypodopaminergia), schizophrenia (mesolimbic hyperdopaminergia hypothesis), and addiction (reward pathway sensitization). Medications: L-DOPA for Parkinson's, stimulants (methylphenidate, amphetamines) for ADHD, antipsychotics (D2 antagonists) for schizophrenia. Natural boosters: exercise, protein-rich diet, adequate sleep, goal-setting activities.
"""
    },
    {
        "title": "Neurotransmitters: Serotonin",
        "text": """
Serotonin (5-HT, 5-hydroxytryptamine) modulates mood, anxiety, sleep, appetite, and social behavior. Synthesized from tryptophan, it acts on 14+ receptor subtypes. Major projections from raphe nuclei to cortex, limbic system, and spinal cord. Implicated in depression (monoamine hypothesis), anxiety disorders, OCD, and eating disorders. SSRIs (selective serotonin reuptake inhibitors) like fluoxetine, sertraline increase synaptic serotonin by blocking reuptake. SNRIs (serotonin-norepinephrine reuptake inhibitors) target both systems. 5-HT2A receptors mediate psychedelic effects. Gut produces 90% of body's serotonin (gut-brain axis). Natural enhancement: tryptophan-rich foods (turkey, eggs, nuts), sunlight exposure, exercise, probiotics. Serotonin syndrome risk with excessive medication combinations.
"""
    },
    {
        "title": "Neurotransmitters: GABA and Glutamate",
        "text": """
GABA (gamma-aminobutyric acid) is the primary inhibitory neurotransmitter, crucial for reducing neuronal excitability and anxiety. Acts on GABA-A (ionotropic, fast inhibition) and GABA-B (metabotropic, slow inhibition) receptors. Benzodiazepines and barbiturates enhance GABA-A function, used for anxiety and seizures. Glutamate is the primary excitatory neurotransmitter, essential for learning, memory via LTP, and neuroplasticity. Acts on ionotropic (AMPA, NMDA, kainate) and metabotropic (mGluR) receptors. NMDA receptor crucial for synaptic plasticity and learning. Excitotoxicity from excessive glutamate causes neuronal damage in stroke, TBI, neurodegenerative diseases. Balance between excitation (glutamate) and inhibition (GABA) critical for brain function. Imbalances linked to epilepsy, anxiety, schizophrenia, autism spectrum disorders.
"""
    },
    {
        "title": "Neurotransmitters: Acetylcholine, Norepinephrine, Others",
        "text": """
Acetylcholine (ACh) mediates attention, learning, memory, and muscle contraction. Cholinergic projections from basal forebrain (nucleus basalis) to cortex and hippocampus. Acts on nicotinic (ionotropic) and muscarinic (metabotropic) receptors. Depletion in Alzheimer's disease; acetylcholinesterase inhibitors (donepezil, rivastigmine) used as treatment. Norepinephrine (noradrenaline) from locus coeruleus modulates arousal, attention, stress response. Implicated in depression, PTSD, ADHD. SNRIs and tricyclic antidepressants increase norepinephrine. Endorphins (endogenous opioids) mediate pain relief, reward, stress response; released during exercise ('runner's high'), laughter, social bonding. Oxytocin ('love hormone') promotes social bonding, trust, empathy, uterine contractions, lactation; potential therapeutic target for autism, social anxiety. Endocannabinoids modulate pain, appetite, mood, memory via CB1/CB2 receptors.
"""
    },
    {
        "title": "Brain Regions: Prefrontal Cortex",
        "text": """
The prefrontal cortex (PFC) is the anterior portion of frontal lobes, critical for executive functions, planning, decision-making, personality, social behavior. Subdivisions: dorsolateral PFC (DLPFC) for working memory, cognitive flexibility, planning; ventromedial PFC (vmPFC) for emotion regulation, reward valuation, moral reasoning; orbitofrontal cortex (OFC) for reward processing, impulse control; anterior cingulate cortex (ACC) for conflict monitoring, error detection, emotion. PFC not fully mature until mid-20s, explaining adolescent risk-taking. Damage causes impulsivity, poor planning, personality changes (Phineas Gage case). Dysfunction in ADHD (hypoactivation), depression (decreased metabolism), schizophrenia (hypofrontality). Treatments: cognitive training, stimulants for ADHD, psychotherapy, TMS targeting DLPFC for depression. Enhanced by adequate sleep, exercise, stress management, cognitive challenges.
"""
    },
    {
        "title": "Brain Regions: Hippocampus and Memory Systems",
        "text": """
Hippocampus in medial temporal lobe is critical for forming new explicit (declarative) memories and spatial navigation. Contains place cells and grid cells for spatial mapping. Subregions: dentate gyrus (neurogenesis site), CA fields (CA1-CA4), subiculum. Consolidation theory: hippocampus temporarily stores memories before cortical consolidation. Patient H.M. (bilateral hippocampal removal) demonstrated anterograde amnesia while preserving procedural memory, confirming multiple memory systems. Vulnerable to stress (cortisol), hypoxia, Alzheimer's disease (early atrophy). Atrophy linked to depression, PTSD, chronic stress. Neurogenesis in dentate gyrus enhanced by exercise, learning, enriched environments. Procedural memory (skills, habits) relies on basal ganglia and cerebellum, spared in hippocampal damage. Working memory uses prefrontal-parietal networks. Emotional memory involves amygdala-hippocampus interaction.
"""
    },
    {
        "title": "Brain Regions: Amygdala and Emotion",
        "text": """
Amygdala is almond-shaped limbic structure crucial for processing emotions, especially fear, threat detection, emotional memory formation. Receives sensory input directly (fast, unconscious route) and via thalamus-cortex (slow, conscious route). Projects to hypothalamus (autonomic responses), periaqueductal gray (freezing), hippocampus (emotional memory), prefrontal cortex (regulation). Central nucleus orchestrates fear responses. Basolateral complex integrates sensory and contextual information. Hyperactivity linked to anxiety disorders, PTSD (failed fear extinction), depression, aggression. Hypoactivity in psychopathy (reduced fear response). Fear conditioning paradigm used to study learning. Extinction learning (safety learning) requires ventromedial PFC to inhibit amygdala. Treatments: exposure therapy (enhances extinction), SSRIs (reduce hyperactivity), propranolol (blocks reconsolidation). Enhanced emotional intelligence involves amygdala-prefrontal balance.
"""
    },
    {
        "title": "Brain Regions: Basal Ganglia and Motor Control",
        "text": """
Basal ganglia are subcortical nuclei (striatum, globus pallidus, substantia nigra, subthalamic nucleus) critical for motor control, habit formation, reward learning, action selection. Direct pathway (Go) facilitates movement via D1 receptors; indirect pathway (NoGo) inhibits movement via D2 receptors. Parkinson's disease results from dopaminergic neuronal loss in substantia nigra pars compacta, causing tremor, rigidity, bradykinesia, postural instability. Treatments: L-DOPA (dopamine precursor), dopamine agonists, MAO-B inhibitors, deep brain stimulation of subthalamic nucleus or globus pallidus interna. Huntington's disease involves striatal degeneration causing chorea, cognitive decline. Obsessive-compulsive disorder linked to cortico-striato-thalamo-cortical loop dysfunction. Tourette syndrome involves basal ganglia hyperactivity. Habit formation transitions from goal-directed (prefrontal-striatal) to habitual (sensorimotor striatum) with practice.
"""
    },
    {
        "title": "Neurological Disorders: ADHD",
        "text": """
Attention-Deficit/Hyperactivity Disorder (ADHD) is neurodevelopmental disorder characterized by inattention, hyperactivity, impulsivity. Prevalence ~5% children, often persists into adulthood. Neurobiological basis: frontostriatal dysfunction, delayed brain maturation, reduced dopamine and norepinephrine signaling in prefrontal cortex. Structural findings: reduced volume in prefrontal cortex, basal ganglia, cerebellum. Three subtypes: predominantly inattentive, predominantly hyperactive-impulsive, combined. Comorbidities: learning disabilities, anxiety, depression, oppositional defiant disorder. Diagnosis via clinical criteria (DSM-5), behavioral assessments. Treatments: stimulants (methylphenidate, amphetamines; enhance dopamine/norepinephrine; 70-80% response), non-stimulants (atomoxetine, guanfacine, clonidine), behavioral therapy, cognitive training, educational accommodations. Natural approaches: exercise, adequate sleep, omega-3 fatty acids, elimination of food additives, mindfulness training, structure and routines.
"""
    },
    {
        "title": "Neurological Disorders: Depression and Anxiety",
        "text": """
Major Depressive Disorder (MDD) involves persistent low mood, anhedonia, cognitive impairment, vegetative symptoms. Neurobiological factors: monoamine deficiency (serotonin, norepinephrine, dopamine), HPA axis dysregulation, hippocampal atrophy, reduced neuroplasticity, inflammation. Brain changes: decreased activity in dorsolateral PFC, increased activity in amygdala and default mode network. Treatments: SSRIs, SNRIs, bupropion (dopamine/norepinephrine), tricyclics, MAO inhibitors, psychotherapy (CBT, IPT), ECT for treatment-resistant cases, TMS, ketamine for rapid relief, exercise (comparable to medication in mild-moderate cases), light therapy for seasonal affective disorder. Anxiety disorders (GAD, panic, social anxiety, phobias, OCD) involve amygdala hyperactivity, reduced prefrontal regulation. Treatments: SSRIs, benzodiazepines (short-term), buspirone, CBT (especially exposure therapy), mindfulness, relaxation techniques. Natural approaches: exercise, omega-3s, adequate sleep, stress management, yoga, meditation, limiting caffeine.
"""
    },
    {
        "title": "Neurological Disorders: Neurodegenerative Diseases",
        "text": """
Alzheimer's Disease (AD) is most common dementia, characterized by progressive memory loss, cognitive decline. Pathology: amyloid-beta plaques, tau neurofibrillary tangles, starting in hippocampus and spreading. Cholinergic deficit from basal forebrain degeneration. Risk factors: age, APOE4 allele, cardiovascular disease. Treatments: acetylcholinesterase inhibitors (donepezil, rivastigmine, galantamine), memantine (NMDA antagonist), new anti-amyloid antibodies (aducanumab, lecanemab). Prevention: cognitive engagement, exercise, Mediterranean diet, social interaction, managing cardiovascular risk. Parkinson's Disease: motor symptoms from dopaminergic loss in substantia nigra; also causes cognitive impairment, depression. Treatments: L-DOPA, dopamine agonists, MAO-B inhibitors, COMT inhibitors, DBS. Multiple Sclerosis: autoimmune demyelination causing varied neurological symptoms; disease-modifying therapies reduce relapses. ALS: motor neuron degeneration; riluzole and edaravone slow progression.
"""
    },
    {
        "title": "Brain Imaging Modalities: Structural and Functional",
        "text": """
fMRI (functional MRI) measures blood-oxygen-level dependent (BOLD) signals as proxy for neural activity. High spatial resolution (~1-3mm), poor temporal resolution (~2s). Used for mapping brain activation during tasks, resting-state connectivity. Limitations: hemodynamic lag, susceptibility artifacts, correlation not causation. EEG (electroencephalography) records scalp electrical potentials from neuronal activity. Excellent temporal resolution (milliseconds), poor spatial resolution. ERP (event-related potentials) are time-locked averages. Used for studying cognitive processing stages, clinical diagnosis (epilepsy, sleep). MEG (magnetoencephalography) measures magnetic fields; better spatial localization than EEG. PET (positron emission tomography) uses radiotracers for metabolism, neurotransmitter receptors, amyloid imaging. Structural MRI: T1-weighted for anatomy, T2/FLAIR for pathology. DTI/DSI: diffusion imaging for white matter tracts. TMS: transcranial magnetic stimulation for causal interventions, depression treatment. fNIRS: functional near-infrared spectroscopy for portable brain imaging.
"""
    },
    {
        "title": "Cognitive Functions: Memory Systems",
        "text": """
Memory systems: Declarative (explicit) memory includes episodic (personal events, hippocampus-dependent) and semantic (facts, knowledge, temporal cortex). Non-declarative (implicit) includes procedural (skills, habits; basal ganglia/cerebellum), priming (perceptual facilitation), classical conditioning (emotional responses, amygdala). Working memory (prefrontal-parietal) holds information temporarily for manipulation. Atkinson-Shiffrin model: sensory → short-term → long-term memory. Consolidation: synaptic (immediate, LTP-dependent) and systems (gradual, hippocampus to cortex). Sleep crucial for consolidation, especially REM for emotional and slow-wave for declarative. Retrieval involves reconstruction, not playback; susceptible to distortion, false memories. Enhancement strategies: spaced repetition, elaborative encoding, retrieval practice, sleep, exercise, mnemonics, reducing interference. Forgetting: decay, interference (proactive/retroactive), retrieval failure. Memory disorders: amnesia (retrograde/anterograde), dementia, dissociative disorders.
"""
    },
    {
        "title": "Cognitive Functions: Attention and Executive Functions",
        "text": """
Attention involves selecting relevant information while filtering distractions. Networks: dorsal attention (goal-directed, top-down; frontal eye fields, intraparietal sulcus), ventral attention (stimulus-driven, bottom-up; temporoparietal junction, ventral frontal cortex), alerting (maintaining vigilance; locus coeruleus, right frontal/parietal). Selective attention (focus on target), divided attention (multitasking; limited capacity), sustained attention (vigilance). Executive functions are high-level cognitive processes: working memory, cognitive flexibility (set-shifting), inhibitory control (response inhibition). Depend on prefrontal cortex, especially DLPFC. Central executive in Baddeley's model coordinates subsystems. Impairments in ADHD, traumatic brain injury, schizophrenia, aging, frontal lobe damage. Enhancement: cognitive training (dual n-back), physical exercise, adequate sleep, reducing multitasking, mindfulness meditation, challenging mental activities. Attention restoration theory: nature exposure improves attentional capacity.
"""
    },
    {
        "title": "Cognitive Functions: Decision-Making and Reward",
        "text": """
Decision-making involves evaluating options, predicting outcomes, selecting actions. Dual-process theory: System 1 (fast, automatic, emotional; amygdala, striatum) vs System 2 (slow, deliberative, rational; prefrontal cortex). Somatic marker hypothesis (Damasio): emotion-based signals guide decisions. Ventromedial PFC integrates emotion and cognition; damage causes poor real-world decisions despite intact intellect. Orbitofrontal cortex encodes reward value, updates predictions. Iowa Gambling Task assesses decision-making under uncertainty. Reward processing involves mesolimbic dopamine pathway (VTA to nucleus accumbens). Prediction error signals learning: positive (better than expected), negative (worse than expected). Temporal discounting: preference for immediate over delayed rewards; steeper in impulsivity, addiction. Risk-taking involves balancing potential rewards and losses; influenced by age, stress, individual differences. Neuroeconomics combines neuroscience and economics to study choice. Biases: framing effect, sunk cost fallacy, loss aversion, anchoring.
"""
    },
    {
        "title": "Neuroplasticity and Brain Health",
        "text": """
Neuroplasticity is the brain's ability to reorganize structure and function in response to experience, learning, injury. Mechanisms: synaptic plasticity (LTP, LTD; changes in connection strength), structural plasticity (dendritic branching, synaptogenesis, neurogenesis in hippocampus), functional reorganization (cortical remapping). Experience-dependent plasticity: enriched environments enhance cognition, brain volume. Critical periods in development for sensory systems, language. Adult neurogenesis in dentate gyrus enhanced by exercise, learning, antidepressants; inhibited by stress, aging. Recovery after stroke depends on reorganization of adjacent cortex, unaffected hemisphere. Constraint-induced movement therapy enhances motor recovery. Brain health optimization: aerobic exercise (increases BDNF, neurogenesis, hippocampal volume), cognitive engagement (learning new skills, 'use it or lose it'), Mediterranean diet (omega-3s, antioxidants), social interaction, adequate sleep (7-9 hours; consolidates memories, clears metabolites), stress management, avoiding smoking and excessive alcohol. Brain training games show limited transfer to real-world cognition; novel, challenging activities more effective.
"""
    },
    {
        "title": "Psychiatric Treatments: Pharmacological",
        "text": """
Antidepressants: SSRIs (fluoxetine, sertraline, escitalopram; first-line, 4-6 weeks onset), SNRIs (venlafaxine, duloxetine; effective for pain, anxiety), bupropion (dopamine/norepinephrine; less sexual side effects), mirtazapine (sedating, increases appetite), tricyclics (amitriptyline; effective but more side effects), MAO inhibitors (phenelzine; dietary restrictions). Anxiolytics: benzodiazepines (diazepam, alprazolam; fast-acting, dependence risk), buspirone (slow onset, no dependence), SSRIs (first-line for chronic anxiety). Antipsychotics: typical (haloperidol; D2 antagonists; extrapyramidal side effects), atypical (risperidone, olanzapine, quetiapine; lower EPS, metabolic side effects; treat schizophrenia, bipolar, augmentation in depression). Mood stabilizers: lithium (gold standard for bipolar; narrow therapeutic window), valproate, lamotrigine, carbamazepine. Stimulants: methylphenidate, amphetamines (ADHD, narcolepsy). Cognitive enhancers: memantine (Alzheimer's), modafinil (wakefulness). Emerging: psychedelics (psilocybin, MDMA) for depression, PTSD; ketamine for rapid antidepressant effects.
"""
    },
    {
        "title": "Psychiatric Treatments: Non-Pharmacological",
        "text": """
Psychotherapy: Cognitive-Behavioral Therapy (CBT; restructuring thoughts, behavioral activation; evidence-based for depression, anxiety, OCD, PTSD, insomnia), Dialectical Behavior Therapy (DBT; mindfulness, emotion regulation; borderline personality disorder), Acceptance and Commitment Therapy (ACT; psychological flexibility), psychodynamic therapy (insight-oriented), interpersonal therapy (relationship focus). Exposure therapy for anxiety disorders, PTSD (prolonged exposure, virtual reality exposure). EMDR (eye movement desensitization and reprocessing) for trauma. Brain stimulation: Electroconvulsive Therapy (ECT; induced seizure; highly effective for severe depression, catatonia), Transcranial Magnetic Stimulation (TMS; magnetic pulses to DLPFC; FDA-approved for depression), transcranial Direct Current Stimulation (tDCS; weak electrical current), Vagus Nerve Stimulation (VNS; implanted device), Deep Brain Stimulation (DBS; electrodes in brain; severe OCD, depression, Parkinson's). Lifestyle interventions: exercise (aerobic and resistance; antidepressant effects), sleep hygiene, nutrition, stress reduction (mindfulness, yoga, meditation), social support, light therapy (SAD).
"""
    },
    {
        "title": "Research Methods and Experimental Design",
        "text": """
Within-subject designs increase power by reducing between-participant variance but risk carryover effects; counterbalancing mitigates order effects. Between-subject designs avoid carryover but require larger samples. Mixed designs combine both. Key threats: confounds, motion artifacts (in fMRI/EEG), physiological noise, multiple comparisons problem. Common analyses: General Linear Model (GLM) for fMRI/EEG, time-frequency analysis, connectivity (functional, effective), machine learning (pattern classification, predictive modeling). Multiple comparison correction: family-wise error rate (Bonferroni, cluster-based permutation), false discovery rate (FDR; more liberal). Effect sizes (Cohen's d, partial eta-squared) and confidence intervals essential. Power analysis determines sample size (typically .80 power for detecting medium effects). Pre-registration reduces publication bias, p-hacking. Open science practices: data/code sharing, preprints, registered reports. Replication crisis in psychology/neuroscience. Causal inference: randomized controlled trials, natural experiments, instrumental variables, lesion studies, TMS.
"""
    },
    {
        "title": "Neuroethics and Responsible Research",
        "text": """
Obtain informed consent, minimize risk, protect privacy (especially with neuroimaging datasets, genetic information). Avoid overinterpreting correlational imaging results as causal ('reverse inference' problem). Report incidental findings policies in MRI studies (unexpected abnormalities). Neuroenhancement ethics: cognitive enhancers (Modafinil, stimulants) in healthy individuals raises fairness, coercion, authenticity concerns. Brain-computer interfaces (BCIs) for paralyzed patients raise autonomy, privacy, agency issues. Neurotechnology in criminal justice: brain-based lie detection, responsibility assessment; concerns about validity, coercion. 'Neuromyths' in education (learning styles, left/right brain) lack scientific support. Dual-use dilemma: research on aggression, deception could be misused. Responsible conduct: transparent reporting, avoiding sensationalism, acknowledging limitations, diversity in samples (WEIRD populations overrepresented), reproducibility, conflicts of interest disclosure. Public engagement and science communication important for informed societal decisions about neurotechnology.
"""
    },
]

# Optional: expanded Q/A seeds to guide retrieval
FAQ = [
    ("What is the temporal resolution of EEG vs fMRI?",
     "EEG: millisecond temporal resolution; fMRI: seconds-level (hemodynamic) temporal resolution."),
    ("How does LTP relate to memory?",
     "LTP (long-term potentiation) is a synaptic plasticity mechanism believed to underlie learning and memory."),
    ("When should I use a within-subject design?",
     "Use within-subject when you want more power and expect stable performance; watch for order/carryover effects and counterbalance."),
    ("What is dopamine's role in the brain?",
     "Dopamine regulates reward, motivation, motor control, and executive functions through mesolimbic, mesocortical, and nigrostriatal pathways."),
    ("How do SSRIs work?",
     "SSRIs (Selective Serotonin Reuptake Inhibitors) block serotonin reuptake, increasing synaptic serotonin availability; used for depression and anxiety."),
    ("What causes Parkinson's disease?",
     "Parkinson's results from dopaminergic neuronal loss in substantia nigra pars compacta, causing motor symptoms like tremor, rigidity, and bradykinesia."),
    ("What is the role of the hippocampus?",
     "Hippocampus is critical for forming new explicit memories and spatial navigation; damage causes anterograde amnesia."),
    ("How does neuroplasticity work?",
     "Neuroplasticity involves synaptic changes (LTP/LTD), structural changes (dendritic growth, neurogenesis), and functional reorganization in response to experience."),
    ("What are the symptoms of ADHD?",
     "ADHD involves inattention, hyperactivity, and impulsivity; caused by frontostriatal dysfunction and reduced dopamine/norepinephrine signaling."),
    ("What treatments exist for depression?",
     "Depression treatments include SSRIs, SNRIs, psychotherapy (CBT), TMS, ECT for severe cases, and lifestyle changes like exercise."),
    ("What is the amygdala's function?",
     "Amygdala processes emotions, especially fear and threat detection; involved in emotional memory formation and autonomic responses."),
    ("How does exercise benefit the brain?",
     "Exercise increases BDNF, promotes neurogenesis, enhances hippocampal volume, improves mood, and has antidepressant effects."),
    ("What is the prefrontal cortex responsible for?",
     "Prefrontal cortex handles executive functions, planning, decision-making, impulse control, working memory, and personality."),
]


# -------------------------------
# 2) Expanded glossary with neurotransmitters, regions, disorders
# -------------------------------
GLOSSARY: Dict[str, str] = {
    # Brain Regions
    "hippocampus": "Medial temporal lobe structure critical for forming new explicit memories and spatial navigation.",
    "amygdala": "Almond-shaped limbic structure processing emotions, especially fear, threat detection, and emotional memory.",
    "prefrontal cortex": "Frontal lobe regions for executive functions, planning, decision-making, impulse control, and working memory.",
    "basal ganglia": "Subcortical nuclei (striatum, globus pallidus, substantia nigra) for motor control, habit formation, and reward learning.",
    "substantia nigra": "Midbrain structure with dopaminergic neurons; degeneration causes Parkinson's disease.",
    "nucleus accumbens": "Ventral striatum structure central to reward processing and motivation; target of mesolimbic dopamine pathway.",
    "ventral tegmental area": "Midbrain region containing dopaminergic neurons; origin of reward pathways; VTA.",
    "anterior cingulate cortex": "Medial frontal region for conflict monitoring, error detection, emotion regulation, and pain processing; ACC.",
    "orbitofrontal cortex": "Ventral prefrontal region for reward valuation, impulse control, and decision-making; OFC.",
    "dlpfc": "Dorsolateral prefrontal cortex: working memory, cognitive flexibility, planning, and executive control.",
    "vmPFC": "Ventromedial prefrontal cortex: emotion regulation, reward valuation, moral reasoning, and decision-making.",
    
    # Neurotransmitters
    "dopamine": "Catecholamine neurotransmitter for reward, motivation, motor control; implicated in Parkinson's, ADHD, addiction, schizophrenia.",
    "serotonin": "Monoamine neurotransmitter (5-HT) modulating mood, anxiety, sleep, appetite; target of SSRIs for depression.",
    "gaba": "Gamma-aminobutyric acid: primary inhibitory neurotransmitter; reduces neuronal excitability; target of benzodiazepines.",
    "glutamate": "Primary excitatory neurotransmitter; crucial for learning, memory (LTP), neuroplasticity; excess causes excitotoxicity.",
    "acetylcholine": "Neurotransmitter for attention, learning, memory, muscle contraction; depleted in Alzheimer's disease.",
    "norepinephrine": "Catecholamine for arousal, attention, stress response; implicated in depression, PTSD, ADHD; noradrenaline.",
    "endorphins": "Endogenous opioids mediating pain relief, reward, stress response; released during exercise and laughter.",
    "oxytocin": "Neuropeptide promoting social bonding, trust, empathy, lactation; 'love hormone'; potential therapeutic for autism.",
    
    # Imaging & Methods
    "bold": "Blood-oxygen-level dependent signal: fMRI measure reflecting relative deoxyhemoglobin changes linked to neural activity.",
    "fmri": "Functional MRI: measures BOLD signals with high spatial (~1-3mm) and poor temporal (~2s) resolution.",
    "eeg": "Electroencephalography: records scalp electrical potentials; excellent temporal (ms), poor spatial resolution.",
    "pet": "Positron emission tomography: uses radiotracers to measure metabolism, neurotransmitter receptors, amyloid plaques.",
    "tms": "Transcranial magnetic stimulation: magnetic pulses create transient brain disruptions; used for research and depression treatment.",
    "meg": "Magnetoencephalography: measures magnetic fields from neural activity; better spatial localization than EEG.",
    "dti": "Diffusion tensor imaging: MRI technique estimating white matter tract integrity and connectivity.",
    "erp": "Event-related potential: time-locked EEG average revealing cognitive processing stages.",
    
    # Plasticity & Memory
    "ltp": "Long-term potentiation: persistent synaptic strength increase after high-frequency stimulation; learning mechanism.",
    "ltd": "Long-term depression: persistent decrease in synaptic strength; complements LTP for learning and memory.",
    "neuroplasticity": "Brain's ability to reorganize structure and function in response to experience, learning, or injury.",
    "neurogenesis": "Formation of new neurons; occurs in adult hippocampal dentate gyrus; enhanced by exercise and learning.",
    "consolidation": "Memory stabilization process; synaptic (immediate) and systems-level (gradual, sleep-dependent) phases.",
    "bdnf": "Brain-derived neurotrophic factor: protein supporting neuron survival, growth, plasticity; increased by exercise.",
    
    # Disorders
    "adhd": "Attention-deficit/hyperactivity disorder: inattention, hyperactivity, impulsivity; frontostriatal dysfunction, dopamine deficits.",
    "parkinson's disease": "Neurodegenerative disorder from substantia nigra dopamine loss; tremor, rigidity, bradykinesia; treated with L-DOPA.",
    "alzheimer's disease": "Most common dementia; amyloid plaques, tau tangles, cholinergic deficit; progressive memory and cognitive loss.",
    "depression": "Major depressive disorder: persistent low mood, anhedonia; monoamine deficiency, hippocampal atrophy; treated with SSRIs, therapy.",
    "schizophrenia": "Psychotic disorder with hallucinations, delusions, cognitive deficits; dopamine hypothesis; treated with antipsychotics.",
    "ptsd": "Post-traumatic stress disorder: intrusive memories, hyperarousal, avoidance after trauma; amygdala hyperactivity; treated with exposure therapy.",
    "ocd": "Obsessive-compulsive disorder: intrusive thoughts (obsessions), repetitive behaviors (compulsions); cortico-striato-thalamic dysfunction.",
    
    # Networks & Concepts
    "default mode network": "Brain network active at rest; includes medial PFC, posterior cingulate; linked to mind-wandering, self-reference, memory.",
    "executive functions": "High-level cognitive processes: working memory, cognitive flexibility, inhibitory control; depend on prefrontal cortex.",
    "working memory": "Temporary storage and manipulation of information; prefrontal-parietal networks; central executive system.",
    "arcuate fasciculus": "White matter tract connecting temporal language regions (Wernicke) to frontal (Broca) for language processing.",
    "hpa axis": "Hypothalamic-pituitary-adrenal axis: stress response system; dysregulation linked to depression, anxiety, PTSD.",
}

# Expanded intent patterns to recognize more query types
INTENT_PATTERNS = {
    "greet": re.compile(r"\b(hi|hello|hey|namaste|yo|greetings)\b", re.I),
    "define": re.compile(r"\b(define|what is|meaning of|explain|tell me about|describe)\b", re.I),
    "compare": re.compile(r"\b(vs\.?|versus|difference between|compare|distinguish|contrast)\b", re.I),
    "design": re.compile(r"\b(experiment|design|within[- ]subject|between[- ]subject|counterbalance|power|study design|methodology)\b", re.I),
    "stats": re.compile(r"\b(glm|anova|regression|multiple comparisons|fdr|permutation|effect size|confidence interval|statistical|analysis)\b", re.I),
    "brain_region": re.compile(r"\b(hippocampus|amygdala|prefrontal|dmn|default mode|broca|wernicke|arcuate|basal ganglia|striatum|cortex|thalamus|cerebellum)\b", re.I),
    "neurotransmitter": re.compile(r"\b(dopamine|serotonin|gaba|glutamate|acetylcholine|norepinephrine|endorphin|oxytocin|neurotransmitter)\b", re.I),
    "disorder": re.compile(r"\b(adhd|parkinson|alzheimer|depression|anxiety|schizophrenia|ptsd|ocd|autism|bipolar|disorder|disease|condition)\b", re.I),
    "treatment": re.compile(r"\b(treatment|therapy|medication|drug|ssri|intervention|cure|help|manage)\b", re.I),
    "imaging": re.compile(r"\b(fmri|eeg|pet|meg|tms|mri|scan|imaging|neuroimaging)\b", re.I),
}

INTENT_PATTERNS = {
    "greet": re.compile(r"\b(hi|hello|hey|namaste|yo)\b", re.I),
    "define": re.compile(r"\b(define|what is|meaning of|explain)\b", re.I),
    "compare": re.compile(r"\b(vs\.?|versus|difference between|compare)\b", re.I),
    "design": re.compile(r"\b(experiment|design|within[- ]subject|between[- ]subject|counterbalance|power)\b", re.I),
    "stats": re.compile(r"\b(glm|anova|regression|multiple comparisons|fdr|permutation|effect size|confidence interval)\b", re.I),
    "brain_region": re.compile(r"\b(hippocampus|amygdala|prefrontal|dmn|default mode|broca|wernicke|arcuate)\b", re.I),
}

# -------------------------------
# 3) Retrieval engine
# -------------------------------
@dataclass
class Retriever:
    docs: List[Dict[str, str]]
    faqs: List[Tuple[str, str]] = field(default_factory=list)
    vectorizer: Optional[TfidfVectorizer] = None
    doc_texts: List[str] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    faq_q_texts: List[str] = field(default_factory=list)
    faq_answers: List[str] = field(default_factory=list)
    matrix_docs: Optional[np.ndarray] = None
    matrix_faqs: Optional[np.ndarray] = None

    def __post_init__(self):
        # Split docs into semi-passages (roughly paragraphs)
        passages, titles = [], []
        for d in self.docs:
            chunks = [c.strip() for c in re.split(r"\n{2,}", d["text"]) if c.strip()]
            for c in chunks:
                passages.append(c)
                titles.append(d["title"])
        self.doc_texts = passages
        self.titles = titles

        self.faq_q_texts = [q for q, _ in self.faqs]
        self.faq_answers = [a for _, a in self.faqs]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        corpus = self.doc_texts + self.faq_q_texts
        tfidf = self.vectorizer.fit_transform(corpus)
        # Partition matrices
        self.matrix_docs = tfidf[: len(self.doc_texts), :]
        self.matrix_faqs = tfidf[len(self.doc_texts) :, :]

    def search(self, query: str, top_k: int = 3):
        qv = self.vectorizer.transform([query])
        scores_docs = cosine_similarity(qv, self.matrix_docs).ravel()
        scores_faqs = cosine_similarity(qv, self.matrix_faqs).ravel() if self.faq_q_texts else np.array([])
        doc_hits = sorted(
            [(i, float(scores_docs[i])) for i in range(len(self.doc_texts))],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        faq_hits = sorted(
            [(i, float(scores_faqs[i])) for i in range(len(self.faq_q_texts))],
            key=lambda x: x[1],
            reverse=True,
        )[: min(top_k, len(self.faq_q_texts))]
        return doc_hits, faq_hits


# -------------------------------
# 4) Assistant core
# -------------------------------
@dataclass
class CognitiveNeuroAssistant:
    retriever: Retriever
    mode: str = "tutor"  # "tutor" | "concise"
    memory: deque = field(default_factory=lambda: deque(maxlen=6))  # short conversation memory

    def set_mode(self, m: str):
        self.mode = "concise" if m.strip().lower().startswith("c") else "tutor"

    def detect_intent(self, query: str) -> Optional[str]:
        for name, pat in INTENT_PATTERNS.items():
            if pat.search(query):
                return name
        return None

    def answer(self, user: str) -> str:
        # Mode switching commands
        if re.match(r"^\s*/mode\s+(tutor|concise)\s*$", user, re.I):
            new_mode = re.findall(r"(tutor|concise)", user, re.I)[0].lower()
            self.set_mode(new_mode)
            return f"Mode set to {self.mode}."

        # Basic greetings
        if INTENT_PATTERNS["greet"].search(user):
            return "Hello! I’m your Cognitive Neuroscience Assistant. Ask me about brain imaging, memory, experimental design, or stats."

        intent = self.detect_intent(user) or "retrieve"

        # Glossary / definitions
        if intent == "define" or intent == "brain_region":
            term = self._extract_term(user)
            if term:
                g = GLOSSARY.get(term.lower())
                if g:
                    return self._format(g)
            # Fall back to retrieval
        if intent in ("compare", "design", "stats", "retrieve", "brain_region", "define"):
            doc_hits, faq_hits = self.retriever.search(user, top_k=3)
            composed = self._compose_answer(user, doc_hits, faq_hits, intent)
            self.memory.append((user, composed))
            return composed

        # Default: retrieve
        doc_hits, faq_hits = self.retriever.search(user, top_k=3)
        composed = self._compose_answer(user, doc_hits, faq_hits, "retrieve")
        self.memory.append((user, composed))
        return composed

    def _extract_term(self, user: str) -> Optional[str]:
        # Try to pull the noun after 'define' / 'what is'
        m = re.search(r"(define|what is|meaning of|explain)\s+([a-zA-Z \-]+)\??", user, re.I)
        if m:
            return m.group(2).strip().lower()
        # Or any glossary term present
        for term in sorted(GLOSSARY.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(term)}\b", user, re.I):
                return term
        return None

    def _compose_answer(
        self,
        user: str,
        doc_hits: List[Tuple[int, float]],
        faq_hits: List[Tuple[int, float]],
        intent: str,
    ) -> str:
        pieces = []

        # If a FAQ is very close, answer that first
        if faq_hits and faq_hits[0][1] > 0.35:
            idx, score = faq_hits[0]
            pieces.append(self._format(self.retriever.faq_answers[idx]))

        # Pull top passages
        for i, score in doc_hits:
            passage = self.retriever.doc_texts[i]
            title = self.retriever.titles[i]
            # Keep it succinct
            snippet = self._clean_snippet(passage, max_chars=550 if self.mode == "tutor" else 280)
            pieces.append(f"{snippet}  — *{title}*")

        # Lightweight, intent-tailored add-ons
        pieces.append(self._intent_tip(intent, user))

        # Deduplicate and join
        uniq = []
        seen = set()
        for p in pieces:
            key = p.strip().lower()
            if key and key not in seen:
                uniq.append(p)
                seen.add(key)

        # Final formatting
        answer = "\n\n".join([p for p in uniq if p.strip()])
        # Add a small footer with guidance
        if self.mode == "tutor":
            answer += "\n\n— Ask me to */mode concise* for shorter answers."
        return answer.strip()

    def _intent_tip(self, intent: str, user: str) -> str:
        if intent == "compare":
            return self._format(
                "Rule of thumb: EEG/MEG → timing; fMRI → spatial maps; combine methods when feasible."
            )
        if intent == "design":
            return self._format(
                "Design tip: Pre-register hypotheses, counterbalance orders, and plan power (a priori)."
            )
        if intent == "stats":
            return self._format(
                "Stats tip: Control family-wise error (e.g., permutation, cluster-wise, or FDR) and report effect sizes."
            )
        if intent == "brain_region":
            return self._format(
                "Remember: regions work in networks; interpret activations within circuit and task context."
            )
        if intent == "neurotransmitter":
            return self._format(
                "Neurotransmitter tip: Consider receptor subtypes, pathways, and interactions with other systems for full understanding."
            )
        if intent == "disorder":
            return self._format(
                "Clinical note: Disorders involve multiple brain systems; treatment often requires multimodal approaches (medication + therapy + lifestyle)."
            )
        if intent == "treatment":
            return self._format(
                "Treatment reminder: Evidence-based approaches vary by individual; consult healthcare professionals for personalized care."
            )
        if intent == "imaging":
            return self._format(
                "Imaging note: Each modality has trade-offs in spatial/temporal resolution; multimodal approaches provide complementary insights."
            )
        if intent == "define":
            return self._format("If you want brief definitions only, try */mode concise*.")
        return ""

    def _format(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        if self.mode == "concise":
            return s
        return textwrap.fill(s, width=88)

    def _clean_snippet(self, text: str, max_chars: int = 400) -> str:
        t = re.sub(r"\s+", " ", text).strip()
        return (t[: max_chars - 1] + "…") if len(t) > max_chars else t


# -------------------------------
# 5) Simple REPL
# -------------------------------
BANNER = """\
Cognitive Neuroscience Assistant 🧠
Type your question. Commands: /mode tutor | /mode concise | /quit
"""

def main():
    retriever = Retriever(KB_DOCS, FAQ)
    bot = CognitiveNeuroAssistant(retriever=retriever, mode="tutor")

    print(BANNER)
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user:
            continue
        if user.lower() in ("/quit", "/exit"):
            print("assistant> Goodbye!")
            break

        reply = bot.answer(user)
        print("assistant> " + reply + "\n")

if __name__ == "__main__":
    main()
