using Pkg

if Sys.iswindows()
    workpath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca\\paper\\rnaseq"
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\Data"
elseif Sys.isunix()
    workpath=ENV["MYSTORAGE"]*"/work/julia/sca/paper/rnaseq"
    datapath=ENV["MYSTORAGE"]*"/work/Data"
end
cd(workpath); Pkg.activate(".")

using AllenBrain, FileIO, Muon

version = "20230630"
manifest = awsmanifest(version)


# WMB-10XV2 Gene expression matrices (data version : 20230630)
expression_matrices = manifest.file_listing["WMB-10Xv2"]["expression_matrices"]
feature_matrix_label = "WMB-10Xv2-HY" # "WMB-10Xv2-TH"(131212×32285)
rpath = expression_matrices[feature_matrix_label]["raw"]["files"]["h5ad"]["relative_path"]
download_base = joinpath(datapath,"AllenBrain")
local_path = joinpath(download_base, split(rpath,"/")... )
AllenBrain.download_dir(manifest, rpath, local_path)
# Load .h5ad file
adata = load(local_path)
adata.X # cells(131212,adata.obs_names) by genes(32285, adata.var_names)

# WMB-10XV3 Gene expression matrices
expression_matrices = manifest.file_listing["WMB-10Xv3"]["expression_matrices"]
feature_matrix_label = "WMB-10Xv3-HPF" # "WMB-10Xv3-MB"(337101 outofmemory,LAPTOP(16G)), "WMB-10Xv3-CTXsp"(78464✕32285)
                                       # "WMB-10Xv3-HY"(162869✕32285), "WMB-10Xv3-HPF"(181055 outofmemory,LAPTOP(16G)),
                                       # "WMB-10Xv3-MY"(192498✕32285,RIS(32G))
rpath = expression_matrices[feature_matrix_label]["raw"]["files"]["h5ad"]["relative_path"]
download_base = joinpath(datapath,"AllenBrain")
local_path = joinpath(download_base, split(rpath,"/")... )
AllenBrain.download_dir(manifest, rpath, local_path)
adata = load(local_path)
adata.x

# WMB-10X annotation data
using CSVFiles, CSV, DataFrames
rpath = manifest.file_listing["WMB-10X"]["metadata"]["cell_metadata_with_cluster_annotation"]["files"]["csv"]["relative_path"]
download_base = joinpath(datapath,"AllenBrain")
local_path = joinpath(download_base, split(rpath,"/")... )
AllenBrain.download_dir(manifest, rpath, local_path)
ldata = CSV.read(local_path, DataFrame)
# choose only for WMB-10Xv2-HY (version : "20231215")
ldata.cell_label # cell_label in cell annotation data
adata.obs_names # cell_label in gene expression data
i = 0
for cl in adata.obs_names
    if cl in ldata.cell_label
        i += 1
        idx = findfirst(s->s==cl,ldata.cell_label)
        @show i, ldata.feature_matrix_label[idx], ldata.cell_label[idx], ldata.cluster_alias[idx]
    else
        @show i, cl
    end
end
# choose only for WMB-10Xv2-HY (version : "20230630" no 'feature_matrix_label' field)
ldata.cell_label # cell_label in cell annotation data
adata.obs_names # cell_label in gene expression data
i = 0
for cl in adata.obs_names
    if cl in ldata.cell_label
        i += 1
        idx = findfirst(s->s==cl,ldata.cell_label)
#        @show i, ldata.library_method[idx], ldata.anatomical_division_label[idx], ldata.cell_label[idx], ldata.cluster_alias[idx]
    else
        @show i, cl
    end
end

ldata.anatomical_division_label[ ldata.anatomical_division_label.=="HY" .&& ldata.library_method .== "10Xv2"]



open(IOSTREAM,,,) begin
    read_chunk(from_file)
    write_chunk(to_file)
end

fp = open(file)
X = mmap(fp)

using AWS: @service
@service S3
# 
aws_config = AWSConfig(; creds=nothing, region="us-east-2") # this region doesn't work
a = S3.list_objects("copernicus-dem-30m/"; aws_config)
aws_config = AWSConfig(; creds=nothing, region="eu-central-1")
a = S3.list_objects("copernicus-dem-30m/"; aws_config)

aws_config = AWSConfig(; creds=nothing, region="us-west-2")
a = S3.list_objects("allen-brain-cell-atlas/"; aws_config)

S3.get_object("allen-brain-cell-atlas",key; aws_config)

s3://allen-brain-cell-atlas/metadata/Zhuang-ABCA-4-CCF/20230830
s3://allen-brain-cell-atlas/expression_matrices/WMB-10Xv2/20230630/WMB-10Xv2-TH-log2.h5ad

julia> manifest.directory_listing
Dict{String, AllenBrain.AWSDirs} with 18 entries:
"Zhuang-ABCA-4-CCF"               => Dict{String, Any}("metadata"=>Dict{String, Any}("ccf_coordinates"=>Dict{String, Any}("files"=>Dict{String, Any}("csv…"
"Zhuang-ABCA-3"                   => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("Zhuang-ABCA-3"=>Dict{String, Any}("raw"=>Dict{String, An…
"Zhuang-ABCA-1-CCF"               => Dict{String, Any}("metadata"=>Dict{String, Any}("ccf_coordinates"=>Dict{String, Any}("files"=>Dict{String, Any}("csv…"
"MERFISH-C57BL6J-638850-sections" => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("C57BL6J-638850.18"=>Dict{String, Any}("raw"=>Dict{String…
"Zhuang-ABCA-1"                   => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("Zhuang-ABCA-1"=>Dict{String, Any}("raw"=>Dict{String, An…
"MERFISH-C57BL6J-638850-CCF"      => Dict{String, Any}("image_volumes"=>Dict{String, Any}("resampled_average_template"=>Dict{String, Any}("files"=>Dict{S…
"Zhuang-ABCA-4"                   => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("Zhuang-ABCA-4"=>Dict{String, Any}("raw"=>Dict{String, An…
"WMB-taxonomy"                    => Dict{String, Any}("metadata"=>Dict{String, Any}("cluster_annotation_term"=>Dict{String, Any}("files"=>Dict{String, A…
"Zhuang-ABCA-2-CCF"               => Dict{String, Any}("metadata"=>Dict{String, Any}("ccf_coordinates"=>Dict{String, Any}("files"=>Dict{String, Any}("csv…"
"Zhuang-ABCA-3-CCF"               => Dict{String, Any}("metadata"=>Dict{String, Any}("ccf_coordinates"=>Dict{String, Any}("files"=>Dict{String, Any}("csv…"
"Allen-CCF-2020"                  => Dict{String, Any}("image_volumes"=>Dict{String, Any}("annotation_boundary_10"=>Dict{String, Any}("files"=>Dict{Strin…
"Zhuang-ABCA-2"                   => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("Zhuang-ABCA-2"=>Dict{String, Any}("raw"=>Dict{String, An…
"WMB-10Xv2"                       => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("WMB-10Xv2-OLF"=>Dict{String, Any}("raw"=>Dict{String, An…
"MERFISH-C57BL6J-638850"          => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("C57BL6J-638850"=>Dict{String, Any}("raw"=>Dict{String, A…
"WMB-10X"                         => Dict{String, Any}("metadata"=>Dict{String, Any}("gene"=>Dict{String, Any}("files"=>Dict{String, Any}("csv"=>Dict{Str…
"WMB-10Xv3"                       => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("WMB-10Xv3-P"=>Dict{String, Any}("raw"=>Dict{String, Any}…
"WMB-neighborhoods"               => Dict{String, Any}("metadata"=>Dict{String, Any}("UMAP20230830-MB-HB-Glut-Sero-Dopa"=>Dict{String, Any}("files"=>Dict…
"WMB-10XMulti"                    => Dict{String, Any}("expression_matrices"=>Dict{String, Any}("WMB-10XMulti"=>Dict{String, Any}("raw"=>Dict{String, Any…

manifest.file_listing["Zhuang-ABCA-4-CCF"]["metadata"]["ccf_coordinates"]["files"]["csv"]["relative_path"]
manifest.file_listing["Zhuang-ABCA-1-CCF"]["metadata"]["ccf_coordinates"]["files"]["csv"]["relative_path"]
manifest.file_listing["Zhuang-ABCA-2-CCF"]["metadata"]["ccf_coordinates"]["files"]["csv"]["relative_path"]

manifest.file_listing["WMB-10XMulti"]["expression_matrices"]["WMB-10XMulti"]["log2"]["files"]["h5ad"]["relative_path"]
manifest.file_listing["WMB-10XMulti"]["expression_matrices"]["WMB-10XMulti"]["raw"]["files"]["h5ad"]["relative_path"]

manifest.file_listing["MERFISH-C57BL6J-638850-sections"]["expression_matrices"]["C57BL6J-638850.01"]["log2"]["files"]["h5ad"]["relative_path"]
                                                                               ["C57BL6J-638850.02"]
                                                                                        ...
                                                                               ["C57BL6J-638850.59"]
manifest.file_listing["Zhuang-ABCA-3"]["expression_matrices"]["Zhuang-ABCA-3"]["log2"]["files"]["h5ad"]["relative_path"]
manifest.file_listing["Zhuang-ABCA-3"]["expression_matrices"]["Zhuang-ABCA-3"]["raw"]["files"]["h5ad"]["relative_path"]
manifest.file_listing["WMB-10Xv2"]["expression_matrices"]["WMB-10Xv2-TH"]["log2"]["files"]["h5ad"]["relative_path"]
manifest.file_listing["WMB-10Xv2"]["expression_matrices"]["WMB-10Xv2-TH"]["raw"]["files"]["h5ad"]["relative_path"]
                                                        ["WMB-10Xv2-OLF"]
                                                            ...
                                                        ["WMB-10Xv2-Isocortex-4"]
manifest.file_listing["Allen-CCF-2020"]["image_volumes"]["annotation_boundary_10"]["files"]["nii.gz"]["relative_path"]
manifest.file_listing["Allen-CCF-2020"]["image_volumes"]["annotation_10"]["files"]["nii.gz"]["relative_path"]
manifest.file_listing["Allen-CCF-2020"]["image_volumes"]["average_template_10"]["files"]["nii.gz"]["relative_path"]


# annotation data
manifest.file_listing["WMB-10X"]["metadata"]["gene"]["files"]["csv"]["relative_path"]


# 10x RNA-seq gene expression data
manifest.file_listing['WMB-10X']['metadata']



# feature_matrix_label => dataset_label	
expressions = Dict{String,String}(
    "WMB-10XMulti"          => "WMB-10XMulti",
    "WMB-10Xv2-CTXsp"       => "WMB-10Xv2",
    "WMB-10Xv2-HPF"         => "WMB-10Xv2",
    "WMB-10Xv2-HY"          => "WMB-10Xv2",
    "WMB-10Xv2-Isocortex-1" => "WMB-10Xv2",
    "WMB-10Xv2-Isocortex-2" => "WMB-10Xv2",
    "WMB-10Xv2-Isocortex-3" => "WMB-10Xv2",
    "WMB-10Xv2-Isocortex-4" => "WMB-10Xv2",
    "WMB-10Xv2-MB"          => "WMB-10Xv2",
    "WMB-10Xv2-OLF"         => "WMB-10Xv2",
    "WMB-10Xv2-TH"          => "WMB-10Xv2",
    "WMB-10Xv3-CB"          => "WMB-10Xv3",
    "WMB-10Xv3-CTXsp"       => "WMB-10Xv3",
    "WMB-10Xv3-HPF"         => "WMB-10Xv3",
    "WMB-10Xv3-HY"          => "WMB-10Xv3",
    "WMB-10Xv3-Isocortex-1" => "WMB-10Xv3",
    "WMB-10Xv3-Isocortex-2" => "WMB-10Xv3",
    "WMB-10Xv3-MB"          => "WMB-10Xv3",
    "WMB-10Xv3-MY"          => "WMB-10Xv3",
    "WMB-10Xv3-OLF"         => "WMB-10Xv3",
    "WMB-10Xv3-P"           => "WMB-10Xv3",
    "WMB-10Xv3-PAL"         => "WMB-10Xv3",
    "WMB-10Xv3-STR"         => "WMB-10Xv3",
    "WMB-10Xv3-TH"          => "WMB-10Xv3"
)






# create AWS accout
# https://aws.amazon.com/getting-started/guides/setup-environment/

# set credential for root user
# - log-in as root user(kdw764:kdw764@gmail.com)
# - On the right side of the navigation bar, choose your account name, and choose Security credentials. If necessary, choose Continue to Security credentials.
# - Choose MFA and follow instruction on the page opened

# Set up users in IAM Identity Center
# https://aws.amazon.com/getting-started/guides/setup-environment/module-two/
# It is considered a security best practice to not use your root account for everyday tasks, but right now you only have a root user. In this tutorial, we will use IAM Identity Center to create an administrative user.
# - log-in as root user
# - In the search bar, enter IAM Identity Center, and then select IAM Identity Center
# - enter 'Enable' on the popup titled 'Enable IAM Identity Center'
# - choose organization type -> if error happen, re login
# - In the IAM identity center, choose group, add group, choose Users, and click add user, assign user to group

# Manage permissions to multiple AWS accounts
# - Choose Dashboard/under Manage permissions to multiple AWS accounts/ choose Manage permissions
# - In the left hand navigation, choose AWS accounts. Under Organizational structure, select the account you created
# - Choose User and groups tab, select Assign users and groups, select one of groups
# - Select permission sets, select Create permission set.
# - For Permissions set type, select Predefined permission set. For Policy for predefined permission set, select AdministratorAccess
# - Specify permission set details, keep the default settings, and choose Next. 
# - Review and create, verify that the Permission set type uses the AWS managed policy AdministratorAccess. Choose Create.
# - Back to Multi-account permissions/Permission sets , push refresh button
# - Select the checkbox for the AdminstratorAccess permission set and select Next.
# - Review and submit, review the selected users and groups and permission set, then choose Submit.

# log-in with IAM username
# Sometimes, it fails. In that case use the link in the invitation email

# Set Up the 
# https://aws.amazon.com/getting-started/guides/setup-environment/module-three/

# Install AWS CLI
# - Install the AWS CLI v2 for your OS, https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
# - C:\> msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
# - relaunch command prompt and check C:\> aws --version

# Configure AWS CLI credentials
# - c:\>aws configure sso
# SSO session name (Recommended): Test1
# SSO start URL [None]: https://d-9a6770a814.awsapps.com/start
# SSO region [None]: us-east-2
# SSO registration scopes [None]: sso:account:access
# You can find URL and region information in the the Settings summary in the IAM Identity Center console Dashboard.
# - log-in on the pop up browser, then go back to command prompt
# The only AWS account available to you is: 654654242601
# Using the account ID 654654242601
# The only role available to you is: AdministratorAccess
# Using the role name "AdministratorAccess"
# CLI default client Region [None]: us-east-2
# CLI default output format [None]: json
# CLI profile name [AdministratorAccess-654654242601]: admin-1
# The suggested profile name is the account ID number followed by an underscore followed by the role name, however for this tutorial, we are going to use a shorter profile name, admin-1.
# This session created a config file located at ~/.aws/config on computers running Linux or macOS, or at C:\Users\ USERNAME \.aws\config on computers running Windows. Your config file will look similar to the example image

# Use this sso-session and profile to request credentials
# aws sso login --profile admin-1

# Set Up Your AWS Cloud9 IDE
# https://aws.amazon.com/getting-started/guides/setup-environment/module-four/
# aws cloud9 create-environment-ec2 --name getting-started --description "Getting started with AWS Cloud9." --instance-type t2.micro --image-id resolve:ssm:/aws/service/cloud9/amis/amazonlinux-1-x86_64/ubuntu-22.04-x86_64 --profile admin-1
# this will return
# {
#     "environmentId": "c296d397100146748316833ce0ed6da3"
# }

# Delete created resources - Optional
# If you don't plan to use the AWS Cloud9 development environment we created in this module, you can delete it by running the following command.
# aws cloud9 delete-environment --environment-id <environmentID> --profile admin-1
# aws cloud9 delete-environment --environment-id c296d397100146748316833ce0ed6da3 --profile admin-1

    
# version = "20231215"
# url = "https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/$(version)/manifest.json"
# using AWS.AWSServices: s3
# manifest = http("GET", url)
# manifest = json.loads(requests.get(url).text)
# print("version: ", manifest['version'])

# using AWS: @service
# @service S3
# # 
# aws_config = AWSConfig(; creds=nothing, region="us-east-1")
# a = S3.copy_object("allen-brain-cell-atlas/expression_matrices/WMB-10Xv2/20230630/WMB-10Xv2-TH-log2.h5ad",allendatapath; aws_config)
