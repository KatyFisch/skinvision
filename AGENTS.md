# SkinVision

Build a hobby computer vision prototype that takes an ordinary phone photo of a
skin lesion and explores which possible diagnoses an image classifier can
suggest. The project is a collaboration between a software developer learning
machine learning and a biostatistics PhD collaborator.

- Use [PLAN.md](PLAN.md) for the current milestones and decisions.
- Start with one clinical close-up per lesion. Treat dermoscopy and optional
  metadata as later experiments with explicit input requirements.
- Keep paired images and other known related samples together when splitting
  data. Document the split, label provenance, and limits of the evaluation.
- Keep raw data in the ignored `data/` directory. Preserve dataset attribution
  and keep patient photos out of Git and notebook outputs committed to Git.
- Keep early predictions in a private research demo using curated examples. The
  project is educational. Any plan to offer predictions on real user photos
  needs independent validation for that use and clinical review.
