from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION

OUT = r"D:\Documents\Autism_Project\PHASE_I_MMASD_PROJECT_REPORT.docx"

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.75)
sec.bottom_margin = Inches(0.75)
sec.left_margin = Inches(0.85)
sec.right_margin = Inches(0.85)

styles = doc.styles
styles['Normal'].font.name = 'Calibri'
styles['Normal'].font.size = Pt(10.5)
styles['Normal'].paragraph_format.space_after = Pt(6)
styles['Normal'].paragraph_format.line_spacing = 1.12
for nm, size in [('Title', 25), ('Heading 1', 17), ('Heading 2', 13), ('Heading 3', 11)]:
    st = styles[nm]
    st.font.name = 'Calibri'
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)

def set_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(str(text)); r.bold = bold; r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(headers, rows, widths=None, font=8.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = 'Table Grid'
    for i,h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i], h, True, (255,255,255), font)
        shade(t.rows[0].cells[i], '233F5D')
    for row in rows:
        cells = t.add_row().cells
        for i,v in enumerate(row):
            set_cell_text(cells[i], v, size=font)
            if len(t.rows) % 2 == 0: shade(cells[i], 'F1F5F8')
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t

def para(text, boldlead=None):
    p = doc.add_paragraph()
    if boldlead and text.startswith(boldlead):
        p.add_run(boldlead).bold = True
        p.add_run(text[len(boldlead):])
    else: p.add_run(text)
    return p

def bullets(items):
    for x in items: doc.add_paragraph(x, style='List Bullet')

def numbered(items):
    for x in items: doc.add_paragraph(x, style='List Number')

# Running header and page number
header = sec.header.paragraphs[0]
header.text = 'PHASE I PROJECT REPORT  |  MMASD ACTIVITY RECOGNITION'
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for r in header.runs:
    r.font.size = Pt(8); r.font.color.rgb = RGBColor(90, 100, 110)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('Phase I  •  ')
fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE'); footer._p.append(fld)

# Cover page
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(100)
r = p.add_run('PHASE I PROJECT REPORT'); r.bold=True; r.font.size=Pt(14); r.font.color.rgb=RGBColor(35,63,93)
p = doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run('MMASD Activity Recognition\nfrom Skeletal Movement Data')
p = doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before=Pt(12)
r=p.add_run('A study of feature-based, graph-temporal, and reliability-aware methods'); r.italic=True; r.font.size=Pt(12)
for _ in range(3): doc.add_paragraph()
for line in ['Student name: ______________________________', 'Register number: ____________________________', 'Department / Institution: _____________________', 'Project guide: _______________________________', 'Academic year: ______________________________']:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run(line).font.size=Pt(11)
doc.add_page_break()

doc.add_heading('Abstract', 1)
para('This Phase I project studies how machine-learning methods can recognize physical therapy activities from skeletal movement data in the MMASD dataset. The work began by reviewing the available data and testing several ways to represent movement, including hand-engineered motion features, graph-based models, temporal models, and combinations of these approaches. The project also examined a separate local 2D OpenPose collection containing 1,374 recordings, ten activity labels, and 32 recording groups. Because the group identifiers have not been verified as participant identities, evaluation results are described as group-held-out rather than participant-independent.')
para('Across the saved MMASD experiments, the strongest result was 82.87% accuracy with a macro-F1 score of 0.7912 in an 11-class feature-based experiment using a multi-bank feature representation and an ensemble of conventional classifiers. This result belongs to a different data configuration from the ten-class 2D OpenPose experiments and should not be compared with them as if they were one benchmark. In the 2D OpenPose Stage 4 experiments, the dynamic bilateral configuration achieved the highest mean fold accuracy at 65.92% (±6.61), with pooled out-of-fold macro-F1 of 0.6402. The full reliability-conditioned model achieved 60.40% mean accuracy (±11.65) and pooled macro-F1 of 0.5947. One nominal ablation duplicated another configuration in the implementation, so it is not counted as an independent confirmation.')
para('Overall, Phase I shows that activity recognition is feasible from the available skeletal data, but performance depends strongly on the representation, class definitions, and evaluation split. The experiments do not establish clinical effectiveness or autism diagnosis. Phase II is planned to investigate 3D data and improved approaches using a more carefully verified evaluation design.')

doc.add_heading('Project Scope and How to Read This Report', 1)
para('This report covers the work completed in Phase I and is limited to MMASD. The project folder contains multiple MMASD data configurations and tasks, so their scores are reported separately. In particular, the 11-class feature-based experiments, the ten-class 2D OpenPose experiments, and exploratory temporal or graph experiments use different inputs and cannot be treated as one direct model comparison. AV-ASD is outside the scope of this report. Phase II plans are described separately from completed Phase I work.')

doc.add_heading('Chapter 1  Introduction', 1)
doc.add_heading('1.1 Background', 2)
para('Autism intervention research can involve observing activities and interactions during therapy. Video and motion data can preserve details about posture and movement across time. For this project, the practical machine-learning task is narrower: classify the physical activity represented in a recorded movement sequence. The model does not diagnose autism, assess a child clinically, or measure treatment outcomes.')
para('A skeleton sequence represents a person using a set of body landmarks, such as shoulders, elbows, wrists, hips, knees, and ankles. It is more compact than raw video and makes movement patterns easier to describe numerically. However, pose estimates can be noisy or incomplete, some activities may look similar, and a model may learn details specific to the recordings rather than the activity itself.')
doc.add_heading('1.2 Problem Addressed', 2)
para('The project addresses the challenge of recognizing therapy-related activities from limited skeletal movement data. It investigates whether hand-designed motion summaries, graph representations of body joints, and temporal learning can distinguish activity classes. It also studies how evaluation should be organized when several recordings share a recording-group identifier and when the mapping from those identifiers to individual participants is unavailable.')
doc.add_heading('1.3 Motivation', 2)
para('The work is motivated by the need to understand which representations are useful for this dataset, rather than assuming that a more complex neural network will perform better. Graph models can represent relationships between connected body parts, while temporal models can learn how movement changes over a sequence. Conventional classifiers can provide strong comparisons when the dataset is modest in size. Comparing these approaches under a consistent split helps reveal their strengths and limitations.')
doc.add_heading('1.4 Objectives', 2)
numbered([
    'Audit the available MMASD data configurations, labels, movement features, and group identifiers.',
    'Implement and compare conventional machine-learning, feature-based, graph-based, and temporal approaches for activity recognition.',
    'Evaluate a 2D pose-based graph-temporal model and test the effect of its selected components through ablation experiments.',
    'Use grouped cross-validation and class-aware metrics to examine performance beyond a single random split.',
    'Document robustness, calibration, reproducibility, and data limitations without making clinical claims.',
    'Define a Phase II direction for 3D data and improved methods based on the evidence from Phase I.'
])
doc.add_heading('1.5 Scope and Boundaries', 2)
para('Phase I includes dataset investigation, preprocessing, the completed MMASD experiments, interpretation of their results, and planning for follow-up work. The main Stage 4 analysis uses local 2D OpenPose sequences. Separate saved experiments use an 11-class feature-based MMASD configuration and other exploratory representations; their results are identified as separate tasks. Phase I does not establish participant-independent generalization because cohort identifiers have not been verified as participant identities. It also does not provide diagnosis, treatment recommendations, or clinical validation.')
doc.add_heading('1.6 Report Organization', 2)
para('Chapter 2 reviews relevant work on therapy datasets and skeleton-based activity recognition. Chapter 3 describes the MMASD data, preprocessing, model families, and evaluation design. Chapter 4 presents the completed experiments and their results. Chapter 5 summarizes the findings, limitations, and planned Phase II work. The appendices list key reproducibility files and the Stage 4 ablation configurations.')

doc.add_heading('Chapter 2  Literature Review and Existing Approaches', 1)
doc.add_heading('2.1 MMASD and Therapy Activity Data', 2)
para('MMASD was introduced as a multimodal dataset for studying autism intervention in therapy settings. Its recordings and associated annotations support research on movement and activities in an intervention context [1]. The project uses locally available processed data derived from MMASD. The local files are not identical in every experiment: one pipeline contains 11 activity classes and stored group identifiers, while the Stage 4 2D OpenPose pipeline contains 1,374 sequences across ten classes. This difference matters because class count, sample set, and feature representation all affect the meaning of a score.')
doc.add_heading('2.2 Skeleton-Based Activity Recognition', 2)
para('Skeleton-based recognition represents each frame as a set of body joints and their coordinates. A sequence of skeletons can then be processed by recurrent networks, temporal convolutions, or graph neural networks. ST-GCN introduced a spatial-temporal graph formulation in which body joints form nodes, anatomical connections form spatial edges, and corresponding joints across frames form temporal links [2]. This gives a model a direct way to learn both body structure and movement over time.')
para('Later work explored adaptive and multi-scale graph relationships. Two-Stream Adaptive GCN learns graph structure from data and uses joint and bone streams [3]. MS-G3D improves spatial and temporal information flow with multi-scale graph operations [4]. CTR-GCN learns channel-specific refinements to the graph topology [5]. These methods show that graph structure can be flexible, but their reported results on large action-recognition datasets do not guarantee the same advantages on a smaller therapy dataset.')
doc.add_heading('2.3 Attention, Reliability, and Temporal Features', 2)
para('Graph attention networks assign different weights to neighboring nodes, allowing the model to emphasize some relationships more than others [6]. For pose-based data, confidence values can also indicate whether a detected joint is likely to be reliable. The project therefore tested confidence-aware graph inputs and reliability-conditioned components. The results must be interpreted empirically: confidence information is not guaranteed to improve recognition, especially if the scores are not well calibrated or if the dataset has limited examples.')
para('Temporal feature engineering offers another approach. Instead of learning directly from every frame, a model can use summary statistics, motion derivatives, temporal bins, and class-prototype distances. Such features can be effective on small datasets, although they may lose detail about when a movement occurred. The project compared these compact representations with graph-temporal models rather than assuming either family would be superior.')
doc.add_heading('2.4 Explainability and Calibration', 2)
para('Attribution methods can help identify which input features influence a model output. Grad-CAM is a widely used gradient-based method for highlighting influential regions in a network [7]. In this project, earlier work explored gradient-based importance over skeletal joint indices and temporal positions. Such visualizations are descriptive: they do not prove that a feature causes a prediction or that it has clinical meaning. Calibration methods, including temperature scaling, can separately assess whether model confidence aligns with empirical correctness [8].')
doc.add_heading('2.5 Research Gap and Project Position', 2)
para('The literature establishes graph-temporal learning as a reasonable approach for skeleton activity recognition, but the best design depends on the data and evaluation protocol. The Phase I contribution is therefore an empirical investigation of multiple representations and model components on the available MMASD configurations. The work reports where graph-temporal approaches performed well and where simpler feature-based models remained stronger. It does not claim a new state-of-the-art result or a validated clinical system.')

doc.add_heading('Chapter 3  Methodology', 1)
doc.add_heading('3.1 Overall Workflow', 2)
para('The project followed a staged workflow: inspect the data and labels; prepare movement features; train baseline and proposed models; evaluate them with grouped folds; inspect errors and robustness; and record the limits of each comparison. This sequence kept data suitability and evaluation design part of the modelling process.')
doc.add_heading('3.2 MMASD Data Configurations', 2)
table(['Configuration','Available data','Task','Evaluation note'],[
    ['Local feature-based MMASD','1,693 recordings; 11 activity labels; stored groups','Supervised activity recognition using temporal/kinematic feature banks','Five-fold stored-group evaluation; identity mapping not independently verified'],
    ['Local 2D OpenPose MMASD','1,374 recordings; 10 labels; 32 filename-derived groups; 64 frames × 25 BODY_25 joints','Activity recognition from 2D coordinates and motion features','Five cohort-grouped folds; groups are not verified individual participants'],
    ['Exploratory temporal windows','7,049 windows from 1,661 recordings; 64 kinematic channels in the saved adaptation','Unsupervised clustering of movement dynamics','Separate exploratory task; clustering metrics are not classification accuracy']
], [1.35,2.25,1.7,2.0], 8)
para('The class lists and supports differ between the local data configurations. Scores from these configurations are therefore reported in separate subsections and are not ranked against each other as though they came from the same test set.')
doc.add_heading('3.3 Preprocessing and Representations', 2)
para('For the Stage 4 2D OpenPose pipeline, each recording is represented by 64 temporally sampled frames and 25 BODY_25 landmarks. The four input channels are horizontal and vertical coordinates and their frame-to-frame motion features. OpenPose confidence values are carried separately for the corresponding joints. Training-fold statistics are used for data-dependent normalization so that held-out fold information does not affect fitting.')
para('For the 11-class experiment, the saved advanced pipeline combined temporal and kinematic feature banks, selected features within the training fold, and added distances to class prototypes computed from training data. Its Random Forest, ExtraTrees, and XGBoost predictions were combined using the saved ensemble weights. This is a different representation and experiment from Stage 4.')
doc.add_heading('3.4 Models and Stage 4 Ablations', 2)
para('Stage 4 evaluates a fixed-topology reference, dynamic bilateral relationships, reliability-conditioned graph processing, temporal smoothing, a phase stream, alternative bilateral joint pairs, and a confusable-class margin loss. Five folds use stored recording cohorts, with seed 42. The implementation and configurations in the repository define the precise settings. The nominal A8 result duplicates A1 in the current implementation and must not be treated as an independent experiment.')
doc.add_heading('3.5 Evaluation Design and Metrics', 2)
para('The principal Stage 4 protocol uses five cohort-grouped folds. Recordings that share a stored group identifier are kept together when forming the folds. The audit found no group overlap between folds and no duplicate sequences under the checked coordinate hashes. Normalization and calibration are fitted without using the outer test fold. Because the group-to-participant mapping is unverified, the report uses “group-held-out” terminology.')
para('Accuracy measures the proportion of correctly classified recordings. Macro-F1 gives equal importance to each class and is useful when class counts are uneven. Expected calibration error (ECE) summarizes the gap between confidence and accuracy across confidence bins. Robustness experiments apply controlled joint or temporal corruption to held-out inputs. Statistical comparisons include fold-level tests and paired out-of-fold tests; results from five folds have limited statistical power, while clip-level tests should be interpreted with awareness that recordings within a cohort may be related.')

doc.add_heading('Chapter 4  Experiments and Results', 1)
doc.add_heading('4.1 Earlier MMASD Investigations', 2)
para('Phase I tested several approaches before the final 2D OpenPose ablation study. In the 11-class stored-group experiment, the saved advanced multi-bank ensemble achieved 82.87% accuracy and 0.7912 macro-F1. It combined 800 selected base features with 11 class-prototype distances and ensembled Random Forest, ExtraTrees, and XGBoost. This is the strongest saved result for that particular 11-class configuration. It is not a result for the ten-class 2D OpenPose experiment, and it is not a clinical assessment result.')
table(['MMASD experiment','Accuracy','Macro-F1','Interpretation'],[
    ['Historical temporal GNN (earlier split)','76.44%','0.7530','Historical result; split differs from later grouped evaluation'],
    ['Nested Random Forest baseline','81.10%','0.7716','Five outer folds with inner group-based selection'],
    ['Weighted RF + ExtraTrees + XGBoost','82.04%','0.7812','Exploratory blend; configuration differed from the final ensemble'],
    ['Advanced multi-bank ensemble (11-class)','82.87%','0.7912','Best saved result in this configuration; multiple pipeline components changed together']
], [2.45,0.9,0.9,3.05], 8.5)
para('The increase from 82.04% to 82.87% is modest and came from a pipeline-level change rather than a controlled one-component ablation. The stored group identifiers still need external confirmation before the result can be described as generalization to unseen participants.')
doc.add_heading('4.2 Unsupervised Temporal Dynamics', 2)
para('A separate exploration adapted a temporal autoencoder and clustering workflow to MMASD movement windows. The local PCA-plus-k-means baseline achieved silhouette 0.0343, activity-label ARI 0.0928, NMI 0.2198, and cluster-seed stability ARI 0.7422. The denoising temporal Transformer autoencoder followed by k-means achieved silhouette 0.1474, ARI 0.2886, NMI 0.4406, and seed-stability ARI 0.6203. The learned clusters were more compact and more associated with activity labels than the PCA baseline, while they were less stable across seeds. These are unsupervised clustering measures, not activity-classification accuracy.')
doc.add_heading('4.3 Quality-Aware Graph Pilot', 2)
para('A graph model over body groups with a bidirectional GRU was tested using different label budgets and synthetic missing-body-group masks. At the full label budget, the supervised quality-mask model achieved 62.33% accuracy and 0.5079 macro-F1 on clean input. At 40% synthetic masking, the self-supervised quality-mask variant achieved 53.95% accuracy and 0.4407 macro-F1. The cache used for this pilot did not contain real joint-confidence or missingness measurements, so the masking study is a controlled synthetic test. This pilot did not exceed the best saved feature-based result and does not prove that the model is robust to real-world tracking errors.')
doc.add_heading('4.4 Stage 4 2D OpenPose Ablation Results', 2)
para('Stage 4 compared ten experiment identifiers using the same five cohort-grouped folds. Mean accuracy is averaged across folds; macro-F1 is calculated from pooled out-of-fold predictions; ECE is averaged across folds. The highest mean accuracy was obtained by A1 and the implementation-duplicate A8. A3 is the full RCDB-STGT configuration.')
table(['ID','Configuration summary','Mean accuracy ± SD','Pooled OOF macro-F1','Mean ECE'],[
    ['A0','Fixed bilateral topology reference','57.12% ± 5.88%','0.5428','0.1370'],
    ['A1','Dynamic bilateral connections; reliability provided to edge learner','65.92% ± 6.61%','0.6402','0.0733'],
    ['A2','Fixed topology with reliability only','57.12% ± 5.88%','0.5428','0.1370'],
    ['A3','Full RCDB-STGT: dynamic connections, reliability gate, temporal smoothing','60.40% ± 11.65%','0.5947','0.1109'],
    ['A4','Full model without temporal smoothing','59.53% ± 4.50%','0.5736','0.1074'],
    ['A5','Full model with phase stream','60.55% ± 6.40%','0.5942','0.0817'],
    ['A6','Full model using wrists-only bilateral pair','61.28% ± 4.26%','0.5872','0.0825'],
    ['A7','Full model using five bilateral pairs','58.95% ± 11.32%','0.5728','0.0919'],
    ['A8','Implementation duplicates A1; not an independent ablation','65.92% ± 6.61%','0.6402','0.0733'],
    ['A9','Full model with confusable-class margin loss','61.22% ± 2.25%','0.5878','0.0838']
], [0.45,3.35,1.25,1.2,0.75], 7.4)
para('A1’s 65.92% is the highest mean fold accuracy among these Stage 4 runs. It is not the highest result across every MMASD experiment, because the separate 11-class feature-based pipeline reached 82.87% under a different task and representation. A8 has the same saved scores as A1 because the implementation instantiates the same effective configuration; it adds no separate evidence.')
doc.add_heading('4.5 Selected Statistical Comparisons', 2)
table(['Comparison','Mean accuracy change','Fold t-test p','McNemar p','Bootstrap 95% CI'],[
    ['A0 → A1','+8.81 percentage points','0.0080','1.01 × 10⁻¹¹','[+6.33, +11.28] pp'],
    ['A0 → A3','+3.28 percentage points','0.5298','0.0153','[+0.73, +5.97] pp'],
    ['A1 → A3','−5.52 percentage points','0.2343','1.0 × 10⁻⁷','[−7.42, −3.49] pp'],
    ['A4 → A3','+0.87 percentage points','0.8795','0.472','[−1.53, +3.35] pp']
], [1.2,1.55,1.0,1.05,2.2], 8)
para('The selected tests indicate a clear difference between A0 and A1 under the saved protocol, while the fold-level comparison between A0 and the full A3 model is not statistically significant with only five folds. Clip-level paired tests and bootstrap intervals provide additional information, but they do not remove possible dependence among recordings from the same cohort. These results support a careful empirical conclusion, not a general claim that one architecture will be better on new participants or other datasets.')
doc.add_heading('4.6 Robustness to Synthetic Sensor Corruption', 2)
table(['Model','Condition','Accuracy','Macro-F1'],[
    ['A0','Clean','57.12%','0.4905'], ['A0','40% random joint dropout','49.63%','0.3946'], ['A0','Both wrists occluded','38.91%','0.2732'],
    ['A1','Clean','65.92%','0.5876'], ['A1','40% random joint dropout','51.11%','0.4135'], ['A1','Both wrists occluded','44.79%','0.3936'],
    ['A3','Clean','60.40%','0.5405'], ['A3','40% random joint dropout','49.86%','0.4293'], ['A3','Both wrists occluded','41.27%','0.3616']
], [0.7,2.3,1.2,1.2], 8.5)
para('A1 had the highest absolute accuracy among the three tested models in the wrist-occlusion condition, while A3 had the highest macro-F1 under 40% random joint dropout. The robustness results therefore depend on the type of corruption and metric. They are synthetic perturbation tests and should not be presented as evidence of performance under actual camera or clinical conditions.')
doc.add_heading('4.7 Main Findings', 2)
bullets([
    'The best saved MMASD result overall is 82.87% accuracy and 0.7912 macro-F1 for the separate 11-class, stored-group feature-based task.',
    'Within the ten-class 2D OpenPose Stage 4 experiment, A1 achieved the highest mean fold accuracy at 65.92%; A8 duplicates A1 and is not an independent run.',
    'The full reliability-conditioned model A3 did not outperform A1 on clean Stage 4 data. Adding components together did not guarantee a gain.',
    'The local 2D graph pilots were weaker than feature-based alternatives in earlier experiments, reinforcing the need for strong conventional baselines.',
    'The dataset group split is useful for reducing overlap between related records, but cannot be called participant-independent without verified identity mapping.',
    'Results from the 11-class, ten-class, and unsupervised tasks answer different questions and must remain separately labelled.'
])

doc.add_heading('Chapter 5  Conclusion and Future Scope', 1)
doc.add_heading('5.1 Conclusion', 2)
para('Phase I investigated MMASD activity recognition using a range of movement representations and learning methods. The work progressed from data inspection and baseline modelling to feature-based ensembles, unsupervised temporal analysis, graph-based pilots, and a controlled 2D OpenPose ablation study. The strongest saved result was produced by a feature-based ensemble on the separate 11-class MMASD configuration. In the 2D OpenPose Stage 4 study, dynamic bilateral modelling reached 65.92% mean accuracy, while the full reliability-conditioned configuration reached 60.40%. These findings show that model complexity alone did not determine performance and that the representation and task definition were important.')
para('The project’s main outcome is a documented set of experiments, comparisons, and data-protocol findings. It does not establish that a graph model is universally superior, that the system is clinically effective, or that it can assess autism. A further limitation is that stored recording groups are not verified as individual participants, so participant-independent generalization remains unproven.')
doc.add_heading('5.2 Phase II Plan', 2)
para('Phase II will extend the work to 3D movement data and investigate improved approaches. The following steps are planned; they are not reported as completed Phase I results:')
numbered([
    'Verify the 3D dataset structure, activity labels, joint definitions, recording groups, and participant metadata before selecting a final benchmark.',
    'Define a consistent evaluation protocol that keeps verified participants or sessions separate across training, validation, and testing.',
    'Establish strong and reproducible baselines using conventional feature models and established skeleton graph-temporal architectures.',
    'Test a small number of clearly motivated improvements, such as reliability handling, bilateral relations, temporal representation, or confidence-aware prediction, one component at a time.',
    'Repeat key comparisons across seeds or folds and report accuracy, macro-F1, class-wise errors, calibration, and computational cost.',
    'Keep all claims within activity recognition and the available dataset; do not infer diagnosis or treatment effectiveness from model scores.'
])
doc.add_heading('5.3 Limitations', 2)
bullets([
    'The local MMASD configurations differ in class count, sample count, input features, and split history; their scores are not directly comparable.',
    'The 2D OpenPose data may contain missing, noisy, or inaccurate joints. Synthetic corruption does not fully reproduce natural pose-estimation errors.',
    'The 32 group identifiers in the 2D dataset are filename-derived and are not verified participant identities.',
    'Class imbalance and cohort-specific class distributions can affect both mean performance and fold-to-fold variation.',
    'Some component studies were exploratory, and selected statistical tests cover only a subset of all possible pairwise comparisons.',
    'Attribution and confidence scores describe aspects of model behavior but do not establish causality, clinical meaning, or diagnostic validity.'
])

doc.add_heading('References', 1)
refs = [
    '[1] J. Li et al., “MMASD: A Multimodal Dataset for Autism Intervention Analysis,” Proceedings of the 25th ACM International Conference on Multimodal Interaction, 2023, pp. 397–405. doi: 10.1145/3577190.3614117.',
    '[2] S. Yan, Y. Xiong, and D. Lin, “Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018. doi: 10.1609/aaai.v32i1.12328.',
    '[3] L. Shi et al., “Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.',
    '[4] Z. Liu et al., “Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 143–152.',
    '[5] Y. Chen et al., “Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition,” Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 13359–13368.',
    '[6] P. Veličković et al., “Graph Attention Networks,” International Conference on Learning Representations, 2018.',
    '[7] R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 618–626.',
    '[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 2017, pp. 1321–1330.',
    '[9] K. He et al., “Deep Residual Learning for Image Recognition,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.',
    '[10] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.'
]
for ref in refs:
    p=doc.add_paragraph(ref)
    p.paragraph_format.left_indent=Inches(0.2)
    p.paragraph_format.first_line_indent=Inches(-0.2)
    p.paragraph_format.space_after=Pt(4)

doc.add_heading('Appendix A  Stage 4 Reproduction Files', 1)
para('The following paths are relative to the project folder D:\\Documents\\Autism_Project. They identify the main saved materials used to review and reproduce Stage 4. The exact command and environment are recorded in the reproduction guide.')
table(['Purpose','Project file'],[
    ['Final Stage 4 metrics and interpretation','research_stage4/STAGE4_FINAL_RESULTS.md'],
    ['Dataset and split audit','research_stage4/DATASET_AND_SPLIT_VALIDATION.md'],
    ['Model and experiment settings','research_stage4/EXPERIMENT_CONFIGURATION.md'],
    ['Statistical comparison details','research_stage4/STATISTICAL_ANALYSIS.md'],
    ['Robustness results','research_stage4/ROBUSTNESS_RESULTS.md'],
    ['Reproduction instructions','research_stage4/REPRODUCIBILITY_GUIDE.md'],
    ['Ablation runner','research_stage4/experiments/run_stage4_ablations.py'],
    ['Final Stage 4 report','research_stage4/reports/STAGE4_FINAL_RESULTS.html'],
    ['MMASD Phase I progress and broader comparisons','results/two_dataset_research_progress_report_2026_11_13.md']
], [2.6,4.5], 8.5)

doc.add_heading('Appendix B  Terminology', 1)
table(['Term','Meaning in this report'],[
    ['Group-held-out','The evaluation keeps recordings with the same stored recording-group ID together across folds.'],
    ['Participant-independent','Reserved for an evaluation where participant identities are verified and participants are separated across folds; this has not been established here.'],
    ['Mean fold accuracy','The arithmetic mean of accuracy across the five outer test folds, reported with the fold standard deviation.'],
    ['Pooled OOF macro-F1','Macro-F1 computed by combining the held-out predictions from all outer folds.'],
    ['Phase I','Completed MMASD dataset investigations and experiments summarized in this report.'],
    ['Phase II','Planned study using 3D data and improved approaches; not included as completed experimental evidence.']
], [1.7,5.4], 8.5)

doc.core_properties.title = 'Phase I MMASD Activity Recognition Project Report'
doc.core_properties.subject = 'MMASD physical therapy activity recognition from skeletal movement data'
doc.core_properties.author = 'Project Report Draft'
doc.save(OUT)
print(OUT)
