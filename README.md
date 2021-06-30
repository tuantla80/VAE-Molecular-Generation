## VAE-Molecular-Generation  
- Ref paper: [Automatic chemical design using a data-driven continuous representation of molecules - 2018](https://arxiv.org/abs/1610.02415)
- Epoch vs. Train loss  
   <img src="https://github.com/tuantla80/VAE-Molecular-Generation/blob/main/test/Epoch%20vs.%20Training%20loss.png" width="500" height="360">  
- Note 1: Generated SMILES from this model is tested with RDkit library to know whether or not it is a valid SMILES.  
        I observed that the percentage of valid generated SMILES from this model is very low (<0.5%)  
        A good generated SMILES for dug discovery is more important than the percentage of valid generated SMILES,  
        but the model itself cannot guarantee that the valid generated SMILES is a good candidate.         
        It indicated the limitation of the model for using in production.  
- Note 2: In the next repository, the VAE model combined with Reinforcement Learning will be implemented and  
          hopefully, it can generate SMILES better than this model.
 
