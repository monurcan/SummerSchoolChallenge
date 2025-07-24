# MultimodalDataChallenge2025
Multimodal data challenge 2025 - a DTU Compute summer school challenge

The goal of this challenge is to develop and evaluate algorithms for multimodal learning and classification. 

This challenge is focused on classifying Danish fungi by using photos of the fungi combined with a set of metadata. These are the main goals of the challenge:
- Achieve the highest F1 score by predicting the true label (taxonID_index) of the fungi in the test and final set.
- Implement different multimodal learning strategies including different ways of using the given metadata.
- Find the set of metadata that gives the largest improvement in classification F1 scores.

## Data

There are 183 different types of fungi found in Denmark in this challenge. That means that the label is an integer in the range [0, 182].

The metadata consists of:
- The date when the photo was taken (day, month, year)
- The latitude of the position of the fungi
- The longitude of the position of the fungi
- The habitat where the fungi was found (a textual description)
- The substrate that the fungi was growing on (a textual description)

There are three sets of data in the challenge:
- **Training** Where the photos are name `fungi_trainXXXXXX.jpg`, where XXXXXXX is a zero-padded integer. For the training set, the label (taxonID_index) is given and you can buy metadata (some is available at the start of the challenge).
- **Testing** Where the photos are name `fungi_testXXXXXX.jpg`, where XXXXXXX is a zero-padded integer. This set is used to give team score during the challenge. All metadata is available and the true label should be predicted.
- **Final** Where the photos are name `fungi_finalXXXXXX.jpg`, where XXXXXXX is a zero-padded integer. This set is used to compute the final challenge score. All metadata is available but and true label should be predicted.

## Challenge web site

The main challenge web site can be found [here](fungi.compute.dtu.dk:8080)

## Preparing for the challenge

Please download the [challenge image data](http://fungi.compute.dtu.dk:8080/downloads/FungiImages.zip) before the challenge. It is 13 GB of data.

## Team logins

At the summer school, you will get a team name and its password, so you can login to the challenge site.

