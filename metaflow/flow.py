from metaflow import FlowSpec, step

class Flow(FlowSpec):
    
    @step
    def start(self):
        self.next(self.load_data)
        
    @step
    def load_data(self):
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow is done!")

if __name__ == "__main__":
    Flow()
