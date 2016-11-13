A neural co-reference resolutor is introduced in this paper by Kevin Clark:
https://cs224d.stanford.edu/reports/ClarkKevin.pdf

A similar implementation will be used in Ada. As all other models in Ada, 
there are no heuristics that dominate decision making functions. A black 
box function is learned to classify when a token is a coref or not. 

This nice feature allows Ada to adapt easily to new data. 

