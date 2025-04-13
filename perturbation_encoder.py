from torch.nn import functional as F
import pandas as pd
import numpy as np
import torch
import csv
import os

class PerturbationEncoder:

    def __init__(self, dataset_id, model_type, model_name):
        self.dataset_id = dataset_id
        self.model_type = model_type
        self.model_name = model_name
        self.root = os.path.dirname(os.path.abspath(__file__))+"/required_file/"

        if 'HUVEC' in self.dataset_id:
            self.sirna_to_gene_df = pd.read_csv(
                self.root+'ThermoFisher-export.csv')
            embedding_file = self.root+'perturbation_embedding_rxrx1.csv'
            with open(embedding_file) as csv_file:
                reader = csv.reader(csv_file)
                self.gene_to_embedding_dic = dict(reader)

        if 'BBBC021' in self.dataset_id:
            self.pert_to_embedding = pd.read_csv("/data/pr/molecule_embeddings_rdkit_BBBC021.csv")
                
        if 'rohban' in self.dataset_id:
            self.pert_to_embedding = pd.read_csv(self.root+
                'perturbation_embedding_rohban.csv',
                header=None)
            self.pert_to_embedding.columns = ['gene_name'] + [str(i) for i in range(512)]
        
        if 'morphodiff' in self.dataset_id:
            self.pert_to_embedding = pd.read_csv('/data/pr/molecule_embeddings_rdkit_BBBC021.csv')
            # self.pert_to_embedding = self.pert_to_embedding.rename(columns={'Unnamed: 0':'compound'})


    def get_embedding_for_rxrx1(self, identifier):
        """Return perturbation embedding based on input perturbation
        identifier.

        Args:
            identifier (str): perturbation identifier

        Raises:
            Exception: If there are no embeddings for the gene associated with
            input perturbation

        Returns:
            embedding (tensor): perturbation embedding
        """

        if identifier == 'EMPTY':
            # create an np array of size 512 and fill it with 1
            embedding = np.ones(512)

        else:
            gene = identifier
            if ';' in gene:
                gene = gene.split(';')[0]

            if gene in self.gene_to_embedding_dic.keys():
                embedding = self.gene_to_embedding_dic[gene]
                embedding = embedding.replace('[', '').replace(']', '')
                embedding = np.fromstring(embedding, dtype=float, sep=' ')
            else:
                print("The "+gene+" gene for "+identifier+" is not in the dictionary.")
                raise Exception("The "+gene+" gene for "+identifier+" is not in the dictionary.")

        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)
        return embedding

    def get_embedding_for_bbbc021(self, identifier):
        """Return gene embedding based on input perturbation id.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): gene embedding
        """

        embedding = self.pert_to_embedding[
            self.pert_to_embedding['compound'] == identifier].values[0][1:]
        assert len(embedding) == self.pert_to_embedding.shape[1] - 1
        embedding = embedding.astype(float)
        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)
        return embedding
        
    def get_embedding_for_morphodiff(self, identifier):
        """Return gene embedding based on input perturbation id.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): gene embedding
        """
        embedding = self.pert_to_embedding[
            self.pert_to_embedding['compound'] == identifier].values[0][1:]
        assert len(embedding) == self.pert_to_embedding.shape[1] - 1
        embedding = embedding.astype(float)
        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)
        return embedding
    
    def get_embedding_for_rohban(self, identifier):
        """Return gene embedding based on input perturbation id.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): gene embedding
        """

        embedding = self.pert_to_embedding[
            self.pert_to_embedding['gene_name'] == identifier].values[0][1:]
        assert len(embedding) == self.pert_to_embedding.shape[1] - 1
        embedding = embedding.astype(float)
        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)
        return embedding

    def standardize_gene_dimension(self, embedding):
        """Standardize gene embedding dimension to (bs, 77, 768).

        Args:
            gene_embedding (tensor): gene embedding generated by scGPT in
            512 dimensions

        Returns:
            padded_tensor (tensor): gene embedding padded with value 1 to have
            dimension (77, 768)"""
        # pad gene_embedding with value 1 to have dimension 768
        # (768 is SD prompt encoding size)
        final_size = 768

        # Calculate the amount of padding on each side
        padding_left = (final_size - len(embedding)) // 2
        padding_right = final_size - len(embedding) - padding_left
        # Pad the tensor
        padded_tensor = F.pad(
            embedding, (padding_left, padding_right), 'constant', 1)
        # replicate padded_tensor to have dimension (bs, 77, 768)
        padded_tensor = padded_tensor.repeat(1, 77, 1)
        return padded_tensor

    def get_gene_embedding(self, identifier):
        """Get gene embedding generated by scGPT based on input sirna id.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): perturbation embedding
        """
        embedding = []

        if self.model_type == 'conditional':

            if 'HUVEC' in self.dataset_id:
                embedding = self.get_embedding_for_rxrx1(identifier)

            elif 'BBBC021' in self.dataset_id:
                embedding = self.get_embedding_for_bbbc021(identifier)
            
            elif 'morphodiff' in self.dataset_id:
                embedding = self.get_embedding_for_morphodiff(identifier)
                
            elif 'rohban' in self.dataset_id:
                embedding = self.get_embedding_for_rohban(identifier)

        elif self.model_type == 'naive':

            embedding = torch.ones(
                (1, 77, 768))

        else:

            raise Exception("Model type not recognized.")

        return embedding


class PerturbationEncoderInference:

    def __init__(self, dataset_id, model_type, model_name):
        self.dataset_id = dataset_id
        self.model_type = model_type
        self.model_name = model_name
        self.root = os.path.dirname(os.path.abspath(__file__))+"/required_file/"

        if 'HUVEC' in self.dataset_id:
            self.sirna_to_gene_df = pd.read_csv(
                self.root+'ThermoFisher-export.csv')
            input_file = self.root+'perturbation_embedding_rxrx1.csv'
            with open(input_file) as csv_file:
                reader = csv.reader(csv_file)
                self.gene_to_embedding_dic = dict(reader)

        if 'BBBC021' in self.dataset_id:
            self.pert_to_embedding = pd.read_csv("/home/pr/MorphoDiff/morphodiff/required_file/perturbation_embedding_bbbc021.csv")
                
        if 'rohban' in self.dataset_id:
            self.pert_to_embedding = pd.read_csv(self.root+
                'perturbation_embedding_rohban.csv', header=None)
            self.pert_to_embedding.columns = ['gene_name'] + [str(i) for i in range(512)]

    def __call__(self, identifier):
        """Get gene embedding for input perturbation with unique identifier.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): perturbation embedding
        """
        embedding = []

        if self.model_name == 'SD':

            if self.model_type == 'conditional':

                if 'HUVEC' in self.dataset_id:
                    embedding = self.get_embedding_for_rxrx1(identifier)

                elif 'BBBC021' in self.dataset_id:
                    embedding = self.get_embedding_for_bbbc021(identifier)
                    
                elif 'rohban' in self.dataset_id:
                    embedding = self.get_embedding_for_rohban(identifier)

            elif self.model_type == 'naive':

                embedding = torch.ones(
                    (1, 77, 768))
            
            assert embedding.shape == (1, 77, 768)

        return embedding
    
    def get_embedding_for_morphodiff(self, identifier):
        """Return gene embedding based on input perturbation id.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): gene embedding
        """
        embedding = self.pert_to_embedding[
            self.pert_to_embedding['compound'] == identifier].values[0][1:]
        assert len(embedding) == self.pert_to_embedding.shape[1] - 1
        embedding = embedding.astype(float)
        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)
        return embedding
    
    def standardize_gene_dimension(self, embedding):
        """Standardize gene embedding dimension to (bs, 77, 768).

        Args:
            gene_embedding (tensor): gene embedding generated by scGPT in
            512 dimensions

        Returns:
            padded_tensor (tensor): gene embedding padded with value 1 to have
            dimension (77, 768)"""
        # pad gene_embedding with value 1 to have dimension 768
        # (768 is SD prompt encoding size)
        final_size = 768

        # Calculate the amount of padding on each side
        padding_left = (final_size - len(embedding)) // 2
        padding_right = final_size - len(embedding) - padding_left
        # Pad the tensor
        padded_tensor = F.pad(
            embedding, (padding_left, padding_right), 'constant', 1)
        # replicate padded_tensor to have dimension (bs, 77, 768)
        padded_tensor = padded_tensor.repeat(1, 77, 1).float()

        return padded_tensor

    def get_embedding_for_rxrx1(self, identifier):
        """Return gene embedding generated by scGPT based on input sirna id.

        Args:
            identifier (str): perturbation identifier

        Raises:
            Exception: If there are no embeddings for the gene associated
            with input sirna

        Returns:
            embedding (tensor): gene embedding generated by scGPT in 512
            dimensions
        """

        if identifier == 'EMPTY':
            embedding = np.ones(512)

        else:
            if ';' in identifier:
                identifier = identifier.split(';')[0]
            gene = identifier
            if ';' in gene:
                gene = gene.split(';')[0]

            if gene in self.gene_to_embedding_dic.keys():
                embedding = self.gene_to_embedding_dic[gene]
                embedding = embedding.replace('[', '').replace(']', '')
                embedding = np.fromstring(embedding, dtype=float, sep=' ')
            else:
                print("The "+gene+" gene for "+identifier+" is not in the dictionary.")
                raise Exception("The "+gene+" gene for "+identifier+" is not in the dictionary.")

        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)

        return embedding

    def get_embedding_for_bbbc021(self, identifier):
        """Return gene embedding based on input perturbation id.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): gene embedding
        """
        embedding = self.pert_to_embedding[
            self.pert_to_embedding['compound'] == identifier].values[0][1:]
        assert len(embedding) == self.pert_to_embedding.shape[1] - 1
        embedding = embedding.astype(float)
        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)

        return embedding
    
    def get_embedding_for_rohban(self, identifier):
        """Return gene embedding based on input perturbation id.

        Args:
            identifier (str): perturbation identifier

        Returns:
            embedding (tensor): gene embedding
        """
        assert identifier in self.pert_to_embedding['gene_name'].values
        embedding = self.pert_to_embedding[
            self.pert_to_embedding['gene_name'] == identifier].values[0][1:]
        assert len(embedding) == self.pert_to_embedding.shape[1] - 1
        embedding = embedding.astype(float)
        embedding = torch.from_numpy(embedding)

        if self.model_name == 'SD':
            embedding = self.standardize_gene_dimension(embedding)

        return embedding

