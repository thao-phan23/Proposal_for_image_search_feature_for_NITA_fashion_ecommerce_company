# Project proposal: SHOP BY IMAGE SEARCH
![image](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/b3ec5d8a-3fa3-4307-8602-9ade36eed20d)
# 1. Overview:

### 1.1.About the company: 
NITA Fashion is one of the biggest e-commerce companies distributing over 400 local and international brands. The products are divided into four main categories: apparel, footwear, accessories, and personal care. Customer experience is the company's core value. We consistently strive to identify problems and devise effective solutions, placing utmost importance on ensuring our customers' satisfaction.

### 1.2.Business problem:
As an E-commerce data scientist, I am recently working on a project proposal to enhance the customer experience during product searches. This initiative is based on insights gathered from a survey conducted in 2022, which involved feedback from 2000 customers regarding the challenges they encounter while searching for products using keywords. The survey findings revealed several valuable points:
- 35% of the customers reported difficulties in locating their desired products amidst the vast array of offerings on the website.
- 42% expressed concerns about spending excessive time on the website without being able to find their target items effectively.
- 22% of the customers expressed the demand for an image search feature

### 1.3.Key business question: 
How to improve customer experience during product searches?

### 1.4.Visual search:

I would like to mention visual search (image search and video search) and how potential it is, especially in the fashion e-commerce market.
- Visual search is a new search type with rising demand, in which:
  - 62% of Millennial and Generation Z customers desire visual search over any other new technology. (2019)
  - 54% of US internet users were excited to have this technology in their shopping experience
    ![vs](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/9c6a4538-94a8-45b9-b380-87a8aabf7935)
    
  - Fashion has been always at the top of visual search
    
    ![fs](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/d7afa89e-2cea-4c73-ba4b-fd4cbbba9ee5)

  - The visual search market is projected to experience substantial growth, with a Compound Annual Growth Rate (CAGR) of approximately 18% anticipated until the year 2028.
    
    ![cagr](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/f8643baa-d004-455b-9ecf-c463bf71da03)

Considering the vast potential of the global visual search market, I believe that adopting visual search is a critical and timely move for our company. Being an early adopter will provide us with competitive advantages and allow us to leverage all of its benefits.

### 1.5.Solution:
Given our company's current capacity and available resources, I propose taking a phased approach, starting with phase 1, which will primarily focus on building an image search feature with 2 functions: image classification and image similarity. This will allow us to build a strong foundation and gradually expand into other areas of visual search in the future.

### 1.6.Stakeholders:
The proposal will be introduced to the Chief Executive Officer and all Department Heads to get feedback and approval for its implementation.

# 2. Data understanding:
![image](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/9a6822de-bdab-4cb6-aa72-421f51733b83)

Our dataset consists of more than 44 thousand images from 143 product types, however, for proposal purposes, I will focus on the top 10 product types with more than 24 thousand images.

# 3. Modeling
### 3.1. Image classification modeling:
We will utilize both classic convolutional neural networks and transfer learning networks to build up the first foundational model for image classification tasks, then compare their performance and select the best one.

#### Convolutional neural networks (CNN):
- The best CNN model achieved a test accuracy of 95%.

  ![best cnn](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/00996d42-b8b4-4ec0-957a-9b12c2197ca1)

#### Transfer learning networks:
We then apply transfer learning networks to build up the models to see which is the best one to utilize.

- The best ResNet50 model has an accuracy of 75%.

    ![resnet50](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/3288828a-104f-4265-a7af-dd3285e5eb3d)

- MobiletNetV2 best model improves the test accuracy to 79%.

    ![mobilenet](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/ffefcb86-9126-4dc5-a870-e2f83f615d6b)

- VGG16 has an impressive test accuracy of 95%.

    ![vgg16](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/bb88caf9-1560-4e16-b95b-0656cdf5200c)


### 3.2. Image similarity modeling:

We will build up a Siamese model with utilizing contrastive loss in order to predict the Euclidean distances of image pairs. The shorter distances indicate stronger similarity.

  ![siamese](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/d54bcb03-a35c-4fdc-81bd-dc65a04fa63f)

Test accuracy first achieved 50% and despite various improvement steps, the model's test accuracy remain at only 50%. Due to the time constraint and computational limitations, we will consider this model as our best model at the moment.

# 4. Model evaluation:

The model's performance will be assessed based on accuracy, which involves calculating the ratio of correct predictions to the total number of predictions.
Besides that, f1-score will be another metric to use to analyze the performance of individual classes in the image classification model.

### 4.1.Image classification model:
The image classification model's performance will be evaluated across the same 50 training epochs.

By comparing the performance of all models, the best CNN model turns out to be the best one regarding the accuracy, and computational time within 50 epochs.

  ![Best model](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/244e9289-ba44-4c26-967d-fed9a8a1690d)

When examining the performance in individual classes, 8 out of 10 them exhibit impressive F1-scores ranging from 96% to 100%. However, casual shoes and sports shoes were exceptions, showing slightly lower scores of 82% and 85%, respectively.

  ![f1-score by class](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/c0adfe19-a2f1-4bd2-a663-a4a1d56b03ae)

The explanation for it is visual ambiguity when the machine cannot distinguish the differences between casual shoes and sports shoes, which limits its ability to learn, hence, resulting in lower f1-scores.
Visual ambiguity can be attributed to both subjective and objective causes. Subjective causes might arise from incorrect master data while subjective causes may be rooted in the design nature, where products share a similar appearance but serve different functions. 

### 4.2.Image similarity model.
The image similarity model will experiment with epochs ranging from 5 to 20 to explore various scenarios.

The best model we have will be the one with 50% of accuracy as no superior alternatives have been identified thus far.

![siamese](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/d54bcb03-a35c-4fdc-81bd-dc65a04fa63f)


# 5. Recommendation:

- Check and clean master data:
    - As a standard practice, prior to utilizing data for input into the model, it is crucial to engage the relevant departments and obtain their confirmation on the integrity of their respective working data. This verification process serves to ensure both model accuracy and the delivery of precise outputs to customers.

- Marketing the new image search feature as a competitive advantage again competitors
    - Make media noise to let every user know that our company has this new feature in order to drive traffic to our website, then ultimately increase sales.
    - Additionally, as we encourage more customers to use the new feature, the more data we will collect for our further analysis.

- Utilize image data for buying references.
    - Buying team can utilize image search analysis for buying references to leverage sales, especially from trendy opportunities.

# 6. Deployment:

Here comes what our new feature will look like:
After the customer uploads their target image, the website will classify which type of product types he or she is looking for and return top 5 similar products.

  ![image](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/113264a4-7a99-4b09-ae09-387791341df5)


# 7. Limitation:
- Due to the limited timeline and computational capacity, my models will run maximum 50 epochs. However, training time was significantly extended, with the longest training model taking more than 3 days.
- Data have classes imbalance
- Siamese model's accuracy achieved only 50%

  # 8. Next steps:
- Improve Siamese model's accuracy
- Apply image classification and image similarity for all product types 






