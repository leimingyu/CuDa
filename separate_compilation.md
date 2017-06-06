### sm 
```bash
# Gencode arguments                                                             
SMS ?= 50 52 60                                                                 
                                                                                
ifeq ($(GENCODE_FLAGS),)                                                        
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))                                        
ifneq ($(HIGHEST_SM),)                                                          
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM) 
endif                                                                           
endif
```
