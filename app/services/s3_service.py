import boto3
import os
import io
from typing import Union
from botocore.exceptions import ClientError

class S3Service:
    @staticmethod
    def _get_s3_client():
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )

    @staticmethod
    def subir_archivo(bucket_name: str, source: Union[str, bytes], object_key: str) -> bool:
        try:
            s3_client = S3Service._get_s3_client()

            extra_args = {'ACL': 'public-read'}

            if isinstance(source, str):
                # Es una ruta de archivo local
                s3_client.upload_file(source, bucket_name, object_key, ExtraArgs=extra_args)
                print(f"Archivo {source} subido a s3://{bucket_name}/{object_key}")
            elif isinstance(source, bytes):
                # Es contenido en memoria
                file_obj = io.BytesIO(source)
                s3_client.upload_fileobj(file_obj, bucket_name, object_key, ExtraArgs=extra_args)
                print(f"Contenido subido a s3://{bucket_name}/{object_key}")
                file_obj.close()
            else:
                raise ValueError("El parámetro 'source' debe ser str (ruta) o bytes (contenido)")
                
            return True
        except ClientError as e:
            print(f"Error al subir archivo a S3: {e}")
            return False
        except Exception as e:
            print(f"Error inesperado: {e}")
            return False
