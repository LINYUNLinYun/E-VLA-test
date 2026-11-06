# One additional B4 balance point

After the completed B4 SM24 result (20.4718 samples/s, VLM161.48ms, AE136.60ms) and the neighboring SM32 short result (19.6593 samples/s, VLM169.07ms, AE110.03ms), VLM is the limiting branch at larger quotas. Returning some AE resources to VLM is now evidence-motivated. The hardware descriptor probe confirms20/108 is a distinct supported split.

Screen B4 AE20/Prefix108 at unchanged lead3. This tests whether slightly longer AE can fit within VLM plus next Encoder while shortening the producer branch. Keep all model math and same-B numerical gates unchanged. Only expand to formal100 groups if the whole completed-request throughput improves. This is one targeted point, not a full SM sweep.
