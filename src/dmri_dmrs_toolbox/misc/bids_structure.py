import os
import re

class BIDSStructure:
    def __init__(self, subj, sess, datatype, description, root, folderlevel, workingdir):
        """
        Initialize the BIDS structure.

        Parameters:
        - subj (str): Subject ID.
        - sess (int): Session number.
        - datatype (str): Data type.
        - description (str): Description for the data.
        - root (str): Root directory path.
        - workingdir (str): Working directory path.
        """
        self.bids_strc = {
            "subject": subj,
            "session": f"ses-{sess:02}",
            "datatype": datatype,
            "root": root,
            "folderlevel": folderlevel,
            "workingdir": workingdir,
        }

        self.set_description(description)

    def set_description(self, description=None):
        """
        Update the description and base_name.

        Parameters:
        - description (str): New description for the data.
        """
        if description:
            self.bids_strc["description"] = description
            self.bids_strc["base_name"] = f'{self.bids_strc["subject"]}_{self.bids_strc["session"]}_{description}_'
        else:
            self.bids_strc["description"] = ""
            self.bids_strc["base_name"] = f'{self.bids_strc["subject"]}_{self.bids_strc["session"]}_'

    def get_path(self,suffix=None):
        """
        Construct and return the full BIDS path.

        Parameters:
        - suffix (str): Optional suffix to add to the path.

        Returns:
        - str: Full BIDS-compliant path.
        """
     
        # Build the parts of the path
        parts = [
            self.bids_strc["root"],
            self.bids_strc["folderlevel"],
            self.bids_strc["workingdir"],
            self.bids_strc["subject"],
            self.bids_strc["session"],
            self.bids_strc["datatype"],
        ]

        if self.bids_strc["description"] != "":
            parts.append(self.bids_strc["description"])
        
        # Join all parts into a valid path
        if suffix:
            parts.append(self.bids_strc["base_name"])
            full_path = os.path.join(os.path.join(*parts) + suffix)
        else:
            full_path = os.path.join(*parts)


        return full_path
    
    def set_param(self, **kwargs):
        """
        Update one or more parameters in the BIDS structure.
    
        Parameters:
        - kwargs: Key-value pairs of parameters to update.
        """
        for key, value in kwargs.items():
            if key in self.bids_strc:
                self.bids_strc[key] = value
            else:
                raise KeyError(f"{key} is not a valid parameter in the BIDS structure.")
                
        if 'description' in kwargs:
             self.set_description(self.bids_strc.get('description'))  # This updates base_name automatically

                
    def get_param(self, key):
        """
        Retrieve the value of a specific parameter in the BIDS structure.
    
        Parameters:
        - key (str): The name of the parameter to retrieve.
    
        Returns:
        - The value of the parameter if it exists.
    
        Raises:
        - KeyError: If the parameter does not exist in the BIDS structure.
        """
        if key in self.bids_strc:
            return self.bids_strc[key]
        else:
            raise KeyError(f"{key} is not a valid parameter in the BIDS structure.")

def create_bids_structure(subj, sess, datatype, root, folderlevel, workingdir, description=None, basename=None):
    return BIDSStructure(
        subj=subj,
        sess=sess,
        datatype=datatype,
        root=root,
        folderlevel=folderlevel,
        workingdir=workingdir,
        description=description,
    )

def find_files_with_pattern(bids_structure, pattern, suffix=None):
    """
    Search the directory structure for files matching a given pattern.
    
    Parameters:
    - bids_structure (BIDSStructure): The BIDS structure containing the information for the search.
    - pattern (str): The regular expression pattern to search for in the filenames.
    - suffix (str, optional): A suffix to append to the base path when searching.
    
    Returns:
    - list: A list of paths that match the given pattern.
    """
    # Get the root path using the get_path method from BIDSStructure
    base_path = bids_structure.get_path()
    
    matching_files = []
    for filename in os.listdir(base_path):
        if pattern in filename:
                matching_files.append(os.path.join(base_path, filename))
    
    return matching_files
