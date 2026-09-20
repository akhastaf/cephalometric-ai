# Integration after model selection — planned, not implemented

The master training checkpoint and ONNX artifact contain **all 29 outputs**, with `aariz29-v1`, source IDs, ordered codes and schema checksum recorded in their manifest. Do not export a 14-channel master or discard extra predictions. The first clinical measurement set remains SNA/SNB/ANB. Supporting 29 outputs does not approve new measurements or diagnoses automatically.

Inspection of the current DentalFlow code identified these necessary later changes:

- Python `app/landmarks.py` and Pydantic `ModelManifest`/`LandmarkResponse` currently allow only 14 codes and at most 14 outputs. Add a versioned 29-code registry and explicit mapping from UIT/LIT to existing U1/L1. Keep soft-tissue N_prime/Pog_prime distinct.
- The existing `probability-heatmaps-v1` decoder expects scores in `[0,1]` and uses heatmap-cell centers. Standard HRNet's linear MSE head is **not** that contract. Introduce a versioned raw-heatmap decoder with documented inverse geometry and separately calibrated confidence, or a reviewed export wrapper. Never clamp arbitrary raw scores and call them probabilities. Preserve all 29 channels through parity tests.
- Nest `ai-client.service.ts` also rejects unknown codes and more than 14 points. Extend transport validation and persistence to retain all 29 original predictions. Keep the current v1 clinical/UI projection explicit; the geometry engine continues to use only approved measurement definitions.
- The existing landmark table has a **14-code CHECK constraint and `varchar(8)` code column**. `Pog_prime` is nine characters. Add a new migration expanding the code domain/length; never edit the already-applied migration. Maintain tenant-scoped foreign keys, immutable AI coordinates, audit history and review invalidation.
- The UI may initially show the established 14-point projection while preserving all 29 in the model/analysis record. Additional displayed points and measurement definitions require explicit labels, clinician correction, reset and validation behavior. Future measurements based on retained points can avoid retraining but still need domain/UI work and clinical review.

After selecting the checkpoint, verify PyTorch/ONNX output equivalence and CPU timing/RSS before local Docker integration. Use the existing signed-storage HTTP service, Nest BullMQ worker and measurement/report infrastructure. Do not add a Python queue listener, database access or a GPU production service.

AI-derived measurements remain clearly unreviewed previews until the doctor corrects/accepts landmarks and explicitly validates the analysis. Final clinical reports retain the review state, model/schema version and original-versus-corrected coordinates. No service/schema changes, migrations, model installation or Railway deployment are included in this review package.
