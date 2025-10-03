from abc import ABC, abstractmethod
import torch
from db_connectors import MinIOConnector, TimescaleConnector, KafkaConnector

class ModelSegmentation(ABC):
    """
    Abstract base class for segmentation models.
    
    This class provides a common interface for different segmentation models
    and handles connections to external services like MinIO, TimescaleDB, and Kafka.
    """

    def __init__(
        self,
        model_path,
        model_name
    ):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name


    # Create the model architecture
    @abstractmethod
    def _create_model(self) -> torch.nn.Module | list[torch.nn.Module]:
        """Create and return the model architecture."""
        pass
    
    @abstractmethod
    def load_model(self):
        pass

    
    def minIOConnection(self, address, port, target, access_key, secret_key):
        """
        Returns a MinIO connector instance.
        This is a placeholder for the actual implementation.
        """
        # Assuming you have a MinIOConnector class defined elsewhere
        self.MinIOconnector = MinIOConnector(
            address=address,
            port=port,
            target=target,
            access_key=access_key,
            secret_key=secret_key,
        )

        self.MinIOconnector.connect()

    def timescaleConnection(
        self, address, port, target, username, password, table_name="predictions"
    ):
        """
        Returns a TimescaleDB connector instance.
        This is a placeholder for the actual implementation.
        """
        # Assuming you have a TimescaleDBConnector class defined elsewhere
        self.TimescaleDBconnector = TimescaleConnector(
            address=address,
            port=port,
            target=target,
            username=username,
            password=password,
        )

        self.table = table_name
        self.TimescaleDBconnector.connect()

    def kafkaConnection(
        self,
        address,
        port,
        topic,
        consumer_group,
        auto_offset_reset="earliest",
        security_protocol="plaintext",
        username=None,
        password=None,
    ):
        """
        Returns a Kafka connector instance.
        This is a placeholder for the actual implementation.
        """
        # Assuming you have a KafkaConnector class defined elsewhere
        self.KafkaConnector = KafkaConnector(
            address=address,  # Replace with actual Kafka address
            port=port,  # Replace with actual Kafka port
            target=topic,  # Replace with actual topic name
            consumer_group=consumer_group,  # Replace with actual consumer group
            auto_offset_reset=auto_offset_reset,
            security_protocol=security_protocol,
            username=username,  # Replace with actual username if needed
            password=password,  # Replace with actual password if needed
        )

        self.KafkaConnector.connect()
    
    # This function will work with quixstreams. It should return a dictionary 
    @abstractmethod
    def predict(self, X) -> dict:
        pass

    def __str__(self):
        return f"ModelSegmentation(name={self.model_name}, model_path={self.model_path}, device={self.device})"
    
    def start_streaming(self):
        
        if not hasattr(self, "KafkaConnector") or not self.KafkaConnector:
            raise ValueError("KafkaConnector is not initialized. Please call kafkaConnection() first.")
        if self.KafkaConnector.sdf_stream is None:
            raise ValueError("KafkaConnector.sdf_stream is not initialized.")
    
        self.KafkaConnector.sdf_stream = self.KafkaConnector.sdf_stream.apply(
            self.predict
        )
        
        # Check if the app is initialized
        if self.KafkaConnector.app is None:
            raise ValueError("KafkaConnector.app is not initialized.")
        
        # Start the streaming application
        try:
            self.KafkaConnector.app.run()
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt in streaming app")
            raise  # Re-raise so the main script can handle cleanup
        finally:
            print("Streaming application stopped")

    def disconnect_all(self):
        """Disconnect from all services"""
        print("Disconnecting from all services...")
        try:
            if hasattr(self, "KafkaConnector") and self.KafkaConnector:
                # print("Disconnecting from Kafka...")
                self.KafkaConnector.disconnect()

            if hasattr(self, "MinIOconnector") and self.MinIOconnector:
                # print("Disconnecting from MinIO...")
                self.MinIOconnector.disconnect()

            if hasattr(self, "TimescaleDBconnector") and self.TimescaleDBconnector:
                # print("Disconnecting from TimescaleDB...")
                self.TimescaleDBconnector.disconnect()

            print("All connections closed successfully.")
        except Exception as e:
            print(f"Error during disconnect: {e}")
