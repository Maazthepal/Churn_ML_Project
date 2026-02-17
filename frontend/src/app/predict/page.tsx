'use client';

import { useState } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Loader2, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const formSchema = z.object({
  gender: z.enum(['Male', 'Female']),
  SeniorCitizen: z.enum(['0', '1']),
  Partner: z.enum(['Yes', 'No']),
  Dependents: z.enum(['Yes', 'No']),
  tenure: z.string().min(1, 'Tenure is required').max(2, 'Max 2 digits'),
  PhoneService: z.enum(['Yes', 'No']),
  MultipleLines: z.enum(['Yes', 'No']),
  InternetService: z.enum(['Fiber optic', 'DSL', 'No']),
  OnlineSecurity: z.enum(['Yes', 'No']),
  OnlineBackup: z.enum(['Yes', 'No']),
  DeviceProtection: z.enum(['Yes', 'No']),
  TechSupport: z.enum(['Yes', 'No']),
  StreamingTV: z.enum(['Yes', 'No']),
  StreamingMovies: z.enum(['Yes', 'No']),
  Contract: z.enum(['Month-to-month', 'One year', 'Two year']),
  PaperlessBilling: z.enum(['Yes', 'No']),
  PaymentMethod: z.enum([
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)',
  ]),
  MonthlyCharges: z.string().min(1, 'Monthly Charges is required'),
  TotalCharges: z.string().min(1, 'Total Charges is required'),
});

type FormValues = z.infer<typeof formSchema>;

export default function PredictPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      gender: 'Male',
      SeniorCitizen: '0',
      Partner: 'No',
      Dependents: 'No',
      tenure: '2',
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'Yes',
      StreamingMovies: 'Yes',
      Contract: 'Month-to-month',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Electronic check',
      MonthlyCharges: '110',
      TotalCharges: '220',
    },
  });

  async function onSubmit(values: FormValues) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams(values as any),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error: ${res.status} - ${errorText}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to get prediction. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container mx-auto py-10 px-4 max-w-4xl">
      <h1 className="text-4xl font-bold text-center mb-10">Customer Churn Prediction</h1>

      <Card className="mb-12 shadow-lg">
        <CardHeader className="pb-6">
          <CardTitle className="text-2xl">Enter Customer Details</CardTitle>
          <CardDescription>Fill out the information below to predict churn risk.</CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-10">
              {/* Section 1: Demographics */}
              <div className="space-y-6">
                <h3 className="text-xl font-semibold border-b pb-2">Demographics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <FormField control={form.control} name="gender" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Gender</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Male">Male</SelectItem>
                          <SelectItem value="Female">Female</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="SeniorCitizen" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Senior Citizen</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="0">No</SelectItem>
                          <SelectItem value="1">Yes</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="Partner" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Has Partner?</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="Dependents" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Has Dependents?</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="tenure" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Tenure (months)</FormLabel>
                      <FormControl>
                        <Input type="number" placeholder="0–72" min="0" max="72" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )} />
                </div>
              </div>

              {/* Section 2: Services */}
              <div className="space-y-6">
                <h3 className="text-xl font-semibold border-b pb-2">Services</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <FormField control={form.control} name="PhoneService" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Phone Service</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="MultipleLines" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Multiple Lines</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="InternetService" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Internet Service</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Fiber optic">Fiber optic</SelectItem>
                          <SelectItem value="DSL">DSL</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="OnlineSecurity" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Online Security</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="OnlineBackup" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Online Backup</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="DeviceProtection" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Device Protection</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="TechSupport" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Tech Support</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="StreamingTV" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Streaming TV</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="StreamingMovies" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Streaming Movies</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />
                </div>
              </div>

              {/* Section 3: Contract & Billing */}
              <div className="space-y-6">
                <h3 className="text-xl font-semibold border-b pb-2">Contract & Billing</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <FormField control={form.control} name="Contract" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Contract Type</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Month-to-month">Month-to-month</SelectItem>
                          <SelectItem value="One year">One year</SelectItem>
                          <SelectItem value="Two year">Two year</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="PaperlessBilling" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Paperless Billing</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="PaymentMethod" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Payment Method</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                        <SelectContent>
                          <SelectItem value="Electronic check">Electronic check</SelectItem>
                          <SelectItem value="Mailed check">Mailed check</SelectItem>
                          <SelectItem value="Bank transfer (automatic)">Bank transfer (automatic)</SelectItem>
                          <SelectItem value="Credit card (automatic)">Credit card (automatic)</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="MonthlyCharges" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Monthly Charges ($)</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" placeholder="e.g. 70.50" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )} />

                  <FormField control={form.control} name="TotalCharges" render={({ field }) => (
                    <FormItem>
                      <FormLabel>Total Charges ($)</FormLabel>
                      <FormControl>
                        <Input type="number" step="0.01" placeholder="e.g. 840.00" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )} />
                </div>
              </div>

              <Button type="submit" className="w-full h-12 text-lg" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  'Predict Churn Risk'
                )}
              </Button>
            </form>
          </Form>
        </CardContent>
      </Card>

      {result && (
        <Card className="mt-12 shadow-xl border-t-4 border-blue-600">
          <CardHeader className="text-center pb-2">
            <CardTitle className="text-3xl">Prediction Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-8">
            <div className="text-center">
              <h2 className={`text-4xl font-extrabold ${result.prediction === 1 ? 'text-red-700' : 'text-green-700'}`}>
                {result.churn}
              </h2>
              <p className="text-2xl mt-4 font-medium">
                Churn Probability: <strong>{result.churn_probability}%</strong>
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm font-medium">
                <span>0%</span>
                <span>100%</span>
              </div>
              <Progress value={result.churn_probability} className="h-5" />
            </div>

            <div className="flex justify-center">
              <span className={`px-10 py-4 rounded-full text-white text-xl font-bold shadow-md ${
                result.risk_level.includes('High') ? 'bg-red-600' :
                result.risk_level.includes('Medium') ? 'bg-yellow-500' :
                'bg-green-600'
              }`}>
                {result.risk_level}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <Alert variant="destructive" className="mt-10">
          <AlertCircle className="h-5 w-5" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}